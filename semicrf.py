import torch
import torch.nn as nn
import warnings
class semiCRF(nn.Module):    
    '''
        This class implemented a semi Markov conditional random field.
        batch_first is default true in this implemetation.
        Args:
            tags_num: Number of tags.
            batch_first: Whether the first dimension corresponds to the size of a minibatch.
            L: the semi length. If L = 1, the semiCRF is equivalent to CRF.
    '''
    def __init__(self, tags_num, L, start_idx = None, end_idx = None, batch_first = True):
        self.init_range = 0.01
        self.huge_number = 100000.0
        if tags_num <= 0:
            raise ValueError('The tags_num should be strictly positive %s' % tags_num)
        if L <= 0:
            raise ValueError('The L should be strictly positive %s' % L)            
        super(semiCRF,self).__init__()
        self.tags_num = tags_num
        self.L = L
        self.__transitions = nn.Parameter(torch.empty(tags_num,tags_num))
        #self.__init_transitions = nn.Parameter(torch.empty(tags_num))
        #self.__stop_transitions = nn.Parameter(torch.empty(tags_num))
        self.batch_first = batch_first
        
        # nn.init.uniform_(self.init_transitions, -1.0*self.init_range, self.init_range)
        # nn.init.uniform_(self.stop_transitions, -1.0*self.init_range, self.init_range)
        nn.init.uniform_(self.transitions, -1.0*self.init_range, self.init_range) 

        if start_idx is not None:
            if start_idx >= 0 and start_idx < tags_num:
                self.__transitions.data[:, start_idx] = -1 * self.huge_number
                #self.__init_transitions.data[start_idx] = self.huge_number
            else:
                raise ValueError('The start_index should between 0 and %s, got %s  '% (tags_num - 1, start_idx))   
          

        if end_idx is not None:
            if end_idx >= 0 and end_idx < tags_num:
                self.__transitions.data[end_idx, :] = -1 * self.huge_number
                #self.__stop_transitions.data[end_idx] = self.huge_number 
            else:
                raise ValueError('The end_index should between 0 and %s, got %s  '% (tags_num - 1, end_idx))  
         
    def forward(self, batch_feat, batch_tag, batch_mask, reduction = "none"):
        """Compute the conditional log likelihood of a sequence of tags given emission scores. This part is usually used as a loss function to optimize.
        Args:
            batch_feat (torch.Tensor): Input feature from the previous layer.
                ``(batch_size, seq_length, L, num_tags)`` if ``batch_first`` is ``True``,
                ``(seq_length, batch_size, L, num_tags)`` if ``batch_first`` is ``False``.

            batch_tag (torch.LongTensor): Sequence of tags tensor of size
                ``(batch_size, seq_length)`` if ``batch_first`` is ``True``,
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``.
            batch_mask (torch.ByteTensor): Mask tensor of the same size with batch_tag.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self.check_dimension(batch_feat, batch_tag=batch_tag, batch_mask=batch_mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError('invalid reduction % reduction')
        if batch_mask is None:
            batch_mask = batch_tag.new_ones(batch_tag.shape, dtype=torch.uint8)

        if not self.batch_first:
            batch_feat = batch_feat.transpose(0, 1)
            batch_tag = batch_tag.transpose(0, 1)
            batch_mask = batch_mask.transpose(0, 1)
        score = self.score_sentence(batch_feat,batch_tag,batch_mask)
        
        partition = self.compute_log_partition(batch_feat,batch_mask)
        
        res =  score - partition
        
        if reduction == 'none':
            return res
        if reduction == 'sum':
            return res.sum()
        if reduction == 'mean':
            return res.mean()
        if reduction == 'token_mean':
           return res.sum() / batch_mask.float().sum()
    def score_sentence(self,batch_feat, batch_tag, batch_mask):      
        batch_size, seq_length, L, hidden_dim = batch_feat.shape 
        score = batch_feat[torch.arange(batch_size), 0, 0, batch_tag[:,0]]# + self.__init_transitions[batch_tag[:,0]]               
        self.last_tag = batch_tag[:,0]
        self.last_last_tag = batch_tag[:,0]
        self.last_step = torch.zeros(self.last_tag.shape, dtype=torch.long).to(batch_feat.device)
        for i in range(1,seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            # print (self.last_step)
            score += batch_feat[torch.arange(batch_size),i,self.last_step,batch_tag[:,i]]*batch_mask[:,i].float() #.view(-1)
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.__transitions[self.last_tag,batch_tag[:,i]]*batch_mask[:,i].float()
            
            self.last_last_tag = self.last_tag
            self.last_tag = torch.where(batch_mask[:,i].byte(), batch_tag[:,i], self.last_tag)

            self.last_step = torch.where(batch_mask[:,i].byte(),self.last_step+1, self.last_step)
            self.last_step = torch.where(self.last_last_tag == self.last_tag, self.last_tag, self.last_tag.new_zeros(self.last_tag.shape))
            if not (self.last_step < L).all():
                warnings.warn('The maximum length of a tag in the training data exceeds the L, try to set a bigger L instead')
                self.last_step = torch.clamp(self.last_step, max = L-1)
            
            
            
        # A trick to find the last non-zero position
        # seq_end = batch_mask.float() + 0.1/seq_length*torch.arange(seq_length).float().unsqueeze(0).to(batch_feat.device)
        # seq_end_point = torch.argmax(seq_end,dim = 1)
        # shape: (batch_size,)
        # last_tags = batch_tag[torch.arange(batch_size),seq_end_point]
        # shape: (batch_size,)
        # score += self.__stop_transitions[last_tags]
        return score

    def compute_log_partition(self,batch_feat,batch_mask):
        batch_size, seq_length, L, hidden_dim = batch_feat.shape
        # Start transition score and first emission; score has size of
        # (batch_size, num_tags, L) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        alphas = batch_feat[:,0,0,:]# + self.__init_transitions  
        alpha_list = []
        alpha_list.append(alphas)
        for i in range(L-1):
            alpha_list.append(alphas.new_ones(alphas.shape)*self.huge_number*(-1))

        
        for i in range(1, seq_length): 
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, L, 1)
            broad_alphas = torch.stack(alpha_list,dim = 2)

            #print (broad_alphas[:,0,0])
            #print (broad_alphas[:,0,1])

            broad_alphas = broad_alphas.unsqueeze(3)
            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, L, num_tags)
            e_scores = batch_feat[:,i,:,:].unsqueeze(1)     
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (1, num_tags ,1 ,num_tags)
            t_scores = self.__transitions[:, :].unsqueeze(1).unsqueeze(0)
            scores = e_scores + t_scores + broad_alphas


            scores = scores.view(scores.shape[0],scores.shape[1]*scores.shape[2],scores.shape[3])
            new_alphas = torch.logsumexp(scores, dim= 1) 

            # print (new_alphas.shape)
            # print (alpha_list[-1].shape)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            alphas = torch.where(batch_mask[:,i].unsqueeze(1), new_alphas, alpha_list[0])

            # If it is full, remove the one that is the first
    
            alpha_list.pop()
            alpha_list.insert(0,alphas)
           
        # End transition score
        # shape: (batch_size, num_tags)
        # alphas += self.__stop_transitions            
        return torch.logsumexp(alphas, dim=1)
      
    def decode(self, batch_feat, batch_mask, mask_idx = 0):
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            batch_feat (torch.Tensor): Input feature from the previous layer.
                ``(batch_size, seq_length, num_tags)`` if ``batch_first`` is ``True``,
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``.
            batch_mask (`~torch.ByteTensor`): Mask tensor of size ``(batch_size, seq_length)``
                if ``batch_first`` is ``True``, ``(seq_length, batch_size)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self.check_dimension(batch_feat, batch_mask=batch_mask)
        if batch_mask is None:
            batch_mask = batch_feat.new_ones(batch_feat.shape[:2], dtype=torch.uint8)

        if not self.batch_first:
            batch_feat = batch_feat.transpose(0, 1)
            batch_mask = batch_mask.transpose(0, 1)

        return self.viterbi_decode(batch_feat, batch_mask, mask_idx)
     
    def viterbi_decode(self,batch_feat,batch_mask, mask_idx):
        # Start transition and first emission
        # shape: (batch_size, num_tags)
        batch_size, seq_length, L, hidden_dim = batch_feat.shape        
        alphas = batch_feat[:,0,0,:]# + self.__init_transitions
        alpha_list = []
        alpha_list.append(alphas) 
        for i in range(L-1):
            alpha_list.append(alphas.new_ones(alphas.shape)*self.huge_number*(-1))
      
        history = []
        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag        
        for i in range(1, seq_length):  
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, L, 1)
            broad_alphas = torch.stack(alpha_list,dim = 2)
            broad_alphas = broad_alphas.unsqueeze(3)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, L, num_tags)
            e_scores = batch_feat[:,i,:,:].unsqueeze(1)
            
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (1, num_tags, 1, num_tags)
            t_scores = self.__transitions[:, :].unsqueeze(1).unsqueeze(0)
            scores = e_scores + t_scores + broad_alphas

            
            # Find the maximum score over all possible current tag
            # Viterbi step
            # shape: (batch_size, num_tags)
            # Here should be:
            # new_alphas,idx = torch.max(scores, dim=(1,2))
            # But unfortunately, pytorch do not allow two dims when do max, which makes it very complicated
            scores = scores.view(scores.shape[0],scores.shape[1]*scores.shape[2],scores.shape[3])
            new_alphas,idx = torch.max(scores, dim=1)
            #idx = idx%self.tags_num , idx//self.tags_num
            idx = idx//L, idx%L
            # Should not happen, just in case
            idx = idx[0], torch.clamp(idx[1], max = i-1)
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            alphas = torch.where(batch_mask[:,i].unsqueeze(1), new_alphas, alpha_list[0])

            # If it is full, remove the one that is the first
            alpha_list.pop()
            alpha_list.insert(0,alphas)
            
            history.append(idx)
            
        # End transition score
        # shape: (batch_size, num_tags)
        # alphas += self.__stop_transitions
        # print (history)
        best_tags_list = []
        
        # A trick to find the last non-zero position
        seq_end = batch_mask.float() + 0.1/seq_length*torch.arange(seq_length).float().unsqueeze(0).to(batch_feat.device)
        seq_end_point = torch.argmax(seq_end,dim = 1)

        for i in range(batch_size):
            _, best_last_tag = alphas[i].max(dim=0)
            best_tags = [best_last_tag.item()]
            b_step = 0
            b_step_list = []
            for p,hist in enumerate(reversed(history)): # must start from the right seq_end_point, or it will be a completely different path from another masked start point
                  #print (p)
                  #print (best_last_tag)
                  if p + seq_end_point[i].item() < seq_length - 1:
                      best_tags.append(best_tags[-1])
                  else:
                      if b_step == 0:
                          b_tags,b_step = hist
                          b_tags,b_step = b_tags[i], b_step[i] 
                          best_last_tag = b_tags[best_tags[-1]]
                          b_step = b_step[best_tags[-1]].item()
                          best_tags.append(best_last_tag.item())
                          b_step_list.append(b_step)
                      else:
                         b_step = b_step - 1
                          

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            b_step_list.reverse() 

            #print (best_tags)
            #print (b_step_list)
            # Recover the tag
            best_tags_1 = best_tags[:len(b_step_list)]
            best_tags_2 = best_tags[len(b_step_list):]

            best_tags_recover = []
            for i,j in zip(best_tags_1,b_step_list):
                best_tags_recover.extend([i]*(j+1))
            best_tags_recover.extend(best_tags_2)

            best_tags_list.append(best_tags_recover)
            #print (best_tags_recover)
        res = torch.LongTensor(best_tags_list).to(batch_feat.device)
        if mask_idx is not None:
            res = res.masked_fill(mask = (~ batch_mask), value = mask_idx)
        return res

      
    @property   
    def transitions(self):  
        return self.__transitions  
    @transitions.setter   
    def transitions(self,transitions):  
        transitions = torch.Tensor(transitions)
        if transitions.shape != (self.tags_num,self.tags_num):
            raise ValueError('The transiton dimension should be %s*%s '% self.transitions.shape)        
        self.__transitions = transitions 
    '''
    @property   
    def init_transitions(self):  
        return self.__init_transitions 
    @init_transitions.setter   
    def init_transitions(self,init_transitions):  
        init_transitions = torch.Tensor(init_transitions)
        if init_transitions.shape != self.tags_num:
            raise ValueError('The init_transitions dimension should be %s '% self.init_transitions.shape)        
        self.__init_transitions = init_transitions 
    @property   
    def stop_transitions(self):  
        return self.__stop_transitions   
    @stop_transitions.setter   
    def stop_transitions(self,transitions):  
        stop_transitions = torch.Tensor(stop_transitions)
        if stop_transitions.shape != self.tags_num:
            raise ValueError('The stop_transiton dimension should be %s*%s '% self.stop_transitions.shape)          
        self.__stop_transitions = stop_transitions 
    '''   
    def __repr__(self):
        return "CRF with %s tags" % self.tags_num
                             
    def check_dimension(self, batch_feat, batch_tag = None, batch_mask = None):
        if batch_feat.dim() != 4:
            raise ValueError("batch_feat should be a 4 dimension tensor, got %s" % batch_feat.dim())
        if batch_feat.size(3) != self.tags_num:
            raise ValueError("Expected last dimension of batch_feat equals num_tags, but got %s and %s respectively." % (self.batch_feat.size(3), self.tags_num))
        if batch_tag is not None:
            if batch_feat.shape[:2] != batch_tag.shape:
                raise ValueError("The input feature and the tagged label dimension do not match, got %s and %s respectively." % (str(tuple(batch_feat.shape[:2])), str(tuple(batch_tag.shape))))
        if batch_mask is not None:
            if batch_feat.shape[:2] != batch_mask.shape:
                raise ValueError("The input feature and the mask dimension do not match, got %s and %s respectively." % (str(tuple(batch_feat.shape[:2])), str(tuple(batch_mask.shape))))
            if self.batch_first:
                on = batch_mask[:, 0].all()
            else:
                on = batch_mask[0].all
            if not on:
                raise ValueError('mask of the first timestep must all be on')
