from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn

from soft_embedding import SoftEmbedding

n_tokens = 10
initialize_from_vocab = True

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      # n_tokens=n_tokens, 
                      prepend_tokens=n_tokens, 
                      append_tokens=n_tokens,
                      initialize_from_vocab=initialize_from_vocab)

# model.set_input_embeddings(s_wte)

# set tokenizer pad token
# inputs = tokenizer("May the force be", return_tensors="pt")
# 
# # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
# # even though it does not matter what you pad input_ids with, it's just to make HF happy
# inputs['input_ids'] = torch.cat([torch.full((1,n_tokens), 50256), inputs['input_ids']], 1)
# inputs['attention_mask'] = torch.cat([torch.full((1,n_tokens), 1), inputs['attention_mask']], 1)
# 
# outputs = model(**inputs)

loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

inputs = tokenizer(['Luke, may the force be with', 'Are tacos food?', 'My least favorite food is'], return_tensors="pt", padding=True)
outputs = tokenizer(['you', 'yes', 'tuna'], return_tensors="pt", padding=True)
test = s_wte(inputs, outputs)
breakpoint()
output = model(inputs_embeds=test['inputs_embeds'], attention_mask=test['attention_mask'])
logits = output['logits']
# do cross entropy loss
logit_flat = logits.view(-1, logits.size(-1))
labels_flat = test['labels'].view(-1)
loss = loss_function(logit_flat, labels_flat)


# apply softmax to logits
logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

batch_size = logprobs.size(0)
# do cross entropy loss where target_mask
loss = loss_function(logprobs, outputs['labels'])

pass