from transformers import PreTrainedModel, PretrainedConfig
from pdb import set_trace as breakpoint
import torch
from torch import nn

class SoftPromptModel(PreTrainedModel):
    def __init__(self,
            bm,
            n_prepend: int = 10, 
            n_append: int = 0, 
            freeze_model: bool = True,
            loss_only_target: bool = True,
            random_range: float = 0.5,
            config: PretrainedConfig = PretrainedConfig(),
        ):
        super(SoftPromptModel, self).__init__(config)
        # super(SoftPromptModel, self).__init__()
        # save model
        self.bm = bm
        self.freeze_model = freeze_model
        if self.freeze_model:
            self.freeze_model_weights()
        # get wte from model
        self.wte = bm.get_input_embeddings()
        # save the number of tokens to prepend and append
        self.n_prepend = n_prepend
        self.n_append = n_append
        # get the embedding dimension
        self.embedding_dim = self.wte.weight.size(1)
        
        # save loss_only_target
        self.loss_only_target = loss_only_target

        # initialize the embedding
        if self.n_prepend > 0:
            self.prepend_embedding = nn.parameter.Parameter(self.initialize_embedding(n_prepend, random_range))
        if n_append > 0:
            self.append_embedding = nn.parameter.Parameter(self.initialize_embedding(n_append, random_range))

    def initialize_embedding(self, 
                             n_tokens: int,
                             random_range: float = 0.5, 
                             ):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        return torch.FloatTensor(n_tokens, self.embedding_dim).uniform_(-random_range, random_range)
    
    def freeze_model_weights(self):
        """freeze model weights
        """
        self.freeze_model = True
        for param in self.bm.parameters():
            param.requires_grad = False
    
    def unfreeze_model_weights(self):
        """unfreeze model weights
        """
        self.freeze_model = False
        for param in self.bm.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            # kwargs=None,
            target_ids=None,
            target_mask=None,
            **kwargs,
        ):
        device = input_ids.device

        # first, get embeddings for input tokens
        inputs_embeds = self.wte(input_ids).to(device)
        # labels are input tokens
        labels = input_ids.clone().to(device).long()

        
        
        # if loss only target, make all labels for inputs -100
        if self.loss_only_target:
            labels = labels.fill_(-100)

        def insert3d(a, b, where=None):
            # insert b into a at index where
            c = torch.zeros(a.size(0), a.size(1) + b.size(1), a.size(2))
            # if where None, insert at beginning
            if where is None:
                where = torch.zeros(a.size(0)).long()
            for index in range(a.size(0)):
                split_index = where[index]
                c[index, :split_index, :] = a[index, :split_index, :]
                c[index, split_index:split_index+b.size(1), :] = b[index, :, :]
                c[index, split_index+b.size(1):, :] = a[index, split_index:, :]
            return c
        def insert2d(a, b, where=None):
            # insert b into a at index where
            c = torch.zeros(a.size(0), a.size(1) + b.size(1))
            # if where None, insert at beginning
            if where is None:
                where = torch.zeros(a.size(0)).long()
            for index in range(a.size(0)):
                split_index = where[index]
                c[index, :split_index] = a[index, :split_index]
                c[index, split_index:split_index+b.size(1)] = b[index, :]
                c[index, split_index+b.size(1):] = a[index, split_index:]
            return c
        
        def get_seq_end_index(mask):
            # get location to append embeddings, which is the last non-eos token
            last_one_index = torch.zeros(mask.size(0), dtype=torch.long)
            # get all places where attention_mask is 1
            inds = torch.nonzero(mask)
            for ind in range(mask.size(0)):
                # get highest value from inds
                highest_ind = inds[inds[:,0] == ind][:,1].max()
                last_one_index[ind] = highest_ind
            # first zero index is one more
            seq_end_index = last_one_index + 1
            return seq_end_index

        batch_size = labels.size(0)
        seq_len = labels.size(1)

        # prepend embeddings
        if self.n_prepend > 0:
            # prepend embeddings
            inputs_embeds = insert3d(inputs_embeds, self.prepend_embedding.repeat(batch_size, 1, 1))
            # prepend attention mask of ones
            attention_mask = insert2d(attention_mask, torch.ones(batch_size, self.n_prepend))
            # prepend labels (default: -100)
            labels = insert2d(labels, torch.full((batch_size, self.n_prepend), -100))

        # get last index of each sequence
        seq_end_index = get_seq_end_index(attention_mask)
        # send to same device as inputs
        seq_end_index = seq_end_index.to(device)

        # append embeddings
        if self.n_append > 0:
            # append embeddings
            inputs_embeds = insert3d(inputs_embeds, self.append_embedding.repeat(batch_size, 1, 1), where=seq_end_index)
            # append attention mask of ones
            attention_mask = insert2d(attention_mask, torch.ones(batch_size, self.n_append), where=seq_end_index)
            # append labels (default: -100)
            labels = insert2d(labels, torch.full((batch_size, self.n_append), -100), where=seq_end_index)
        
        target_start_index = seq_end_index + self.n_append
        
        if target_ids is not None:
            # get embeddings for target tokens
            target_embeds = self.wte(target_ids)
            # get embeddings for target tokens
            inputs_embeds = insert3d(inputs_embeds, target_embeds, where=target_start_index)
            # append target_mask
            attention_mask = insert2d(attention_mask, target_mask, where=target_start_index)
            # append target_ids to labels
            labels = insert2d(labels, target_ids, where=target_start_index)
        
        # make sure attention mask and labels are both long
        attention_mask = attention_mask.long()
        labels = labels.long()

        # run through base model
        output = self.bm(
            inputs_embeds=inputs_embeds.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
            # **kwargs, TODO - include kwargs?
        )

        # set target index
        output['target_index'] = target_start_index
        # # shift labels over 1 to offset off by one error
        # new_labels = torch.full_like(labels, -100).to(device).long()
        # new_labels[:, 1:] = labels[:, :-1]
        # labels = new_labels
        output['labels'] = labels

        return output