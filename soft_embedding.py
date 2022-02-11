import torch
import torch.nn as nn
from pdb import set_trace as breakpoint

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                prepend_tokens: int = 10, 
                append_tokens: int = 0, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            prepend_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.prepend_tokens = prepend_tokens
        self.append_tokens = append_tokens
        self.prepend_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               prepend_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
        if append_tokens > 0:
            self.append_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                append_tokens, 
                                                                                random_range, 
                                                                                initialize_from_vocab))
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             prepend_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:prepend_tokens].clone().detach()
        return torch.FloatTensor(prepend_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, inputs):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        def insert3d(a, b, where):
            # insert b into a at index where
            c = torch.zeros(a.size(0), a.size(1) + b.size(1), a.size(2))
            for index in range(a.size(0)):
                split_index = where[index]
                c[index, :split_index, :] = a[index, :split_index, :]
                c[index, split_index:split_index+b.size(1), :] = b[index, :, :]
                c[index, split_index+b.size(1):, :] = a[index, split_index:, :]
            return c
        def insert2d(a, b, where):
            # insert b into a at index where
            c = torch.zeros(a.size(0), a.size(1) + b.size(1))
            for index in range(a.size(0)):
                split_index = where[index]
                c[index, :split_index] = a[index, :split_index]
                c[index, split_index:split_index+b.size(1)] = b[index, :]
                c[index, split_index+b.size(1):] = a[index, split_index:]
            return c

        tokens, attention_mask = inputs['input_ids'], inputs['attention_mask']
        device = tokens.device
        targets = inputs['target_present']
        if targets:
            target_tokens, target_attention_mask = inputs['target_input_ids'], inputs['target_attention_mask']
        # TODO - why???
        # input_embedding = self.wte(tokens[:, self.prepend_tokens:])
        input_embedding = self.wte(tokens)
        prepend_embedding = self.prepend_embedding.repeat(input_embedding.size(0), 1, 1)
        embedding = torch.cat([prepend_embedding, input_embedding], 1)
        attention_mask = torch.cat([torch.ones(input_embedding.size(0), self.prepend_tokens).long().to(device), attention_mask], 1)
        # get index of the last 1 in each row of attention_mask
        last_one_index = torch.max(1-attention_mask, 1)[1]
        # first zero index is one more
        split = last_one_index + 1
        # if targets, insert at last_one_index + 1
        if targets:
            # target_tokens, target_attention_mask = targets['input_ids'], targets['attention_mask']
            target_embedding = self.wte(target_tokens)
            # insert
            embedding = insert3d(embedding, target_embedding, split)
            attention_mask = insert2d(attention_mask, target_attention_mask, split)
        if self.append_embedding is not None:
            # insert
            append_embedding = self.append_embedding.repeat(input_embedding.size(0), 1, 1)
            embedding = insert3d(embedding, append_embedding, split)
            attention_mask = insert2d(attention_mask, torch.ones(input_embedding.size(0), self.append_tokens).long().to(device), split)
        # make a target mask that is zeros everywhere except for the target's attention mask
        target_mask = torch.zeros(attention_mask.size(0), attention_mask.size(1))
        for index, split_ind in enumerate(split):
            # add append tokens so we get to the original index where target starts
            split_ind += self.append_tokens
            if targets:
                target_size = target_attention_mask.size(1)
                target_mask[index, split_ind:split_ind+target_size] = target_attention_mask[index]
            else:
                target_mask[index, split_ind] = 1
        # make lables, eos tokenizer everywhere except for the target
        labels = torch.ones(attention_mask.size(0), attention_mask.size(1)).long().to(device) * -1
        for index, split_ind in enumerate(split):
            target_len = target_attention_mask.size(1)
            inds = torch.where(target_mask[index] == 1)[0].to(device)
            target_inds = inds - split_ind
            # and set
            labels[index, inds] = target_tokens[index, target_inds]

        # TODO - correct?
        # to fix off by one error for targets
        new_target_mask = torch.zeros(target_mask.size(0), target_mask.size(1)).long().to(device)
        new_target_mask[:, 1:] = target_mask[:, :-1]
        target_mask = new_target_mask
        # same for target_tokens
        new_target_tokens = torch.zeros(target_tokens.size(0), target_tokens.size(1)).long().to(device)
        new_target_tokens[:, :-1] = target_tokens[:, 1:]
        # TODO - EOS?
        new_target_tokens[:, -1] = -1
        target_tokens = new_target_tokens
        # and labels
        new_labels = torch.zeros(labels.size(0), labels.size(1)).long().to(device)
        new_labels[:, :-1] = labels[:, 1:]
        new_labels[:, -1] = -1
        labels = new_labels



        if targets:
            inputs = {
                'inputs_embeds': embedding,
                'attention_mask': attention_mask,
                'target_mask': target_mask,
                # 'target_ids': targets['input_ids'],
                'target_ids': target_tokens,
                'split': split,
                'labels': labels,
            }
        else:
            inputs = {
                'inputs_embeds': embedding,
                'attention_mask': attention_mask,
                'target_mask': target_mask,
                'split': split,
                'labels': labels,
            }
        # return torch.cat([prepend_embedding, input_embedding], 1)
        return inputs
