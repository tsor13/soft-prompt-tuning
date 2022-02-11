from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from datasets import load_dataset
from pdb import set_trace as breakpoint

device = 'cuda'
# device = 'cpu'

# load imdb
dataset = load_dataset('imdb')

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# preprocessing
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)
def label_function(examples):
    target = 'positive' if examples['label'] == 1 else 'negative'
    # target = 'bear' if examples['label'] == 1 else 'apple'
    # randomly choose the target
    # target = 'positive' if torch.rand(1) > 0.5 else 'negative'
    output = tokenizer(target, return_tensors='pt')
    return {'target': target, 'target_input_ids': output['input_ids'], 'target_attention_mask': output['attention_mask']}

train_dataset = dataset['test'].map(preprocess_function, batched=False)
train_dataset = train_dataset.map(label_function, batched=False)


from soft_embedding import SoftEmbedding

# model
n_tokens = 10
initialize_from_vocab = True

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      # n_tokens=n_tokens, 
                      prepend_tokens=n_tokens, 
                      append_tokens=n_tokens,
                      initialize_from_vocab=initialize_from_vocab)

# loss function
loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

# data loader
batch_size = 4
# make collocator function that fills in the correct padding
def collocator(batch):
    # for input_ids, pad with tokenizer.pad_token
    # get max length of input_ids
    max_length = max([len(x['input_ids']) for x in batch])
    # pad with pad_token
    pad_token = tokenizer.pad_token_id
    input_ids = torch.full((batch_size, max_length), pad_token).long()
    # fill in with batch['input_ids']
    for i, x in enumerate(batch):
        input_ids[i, :len(x['input_ids'])] = torch.Tensor(x['input_ids']).long()
    
    # do same for attention_mask, but pad with 0
    attention_mask = torch.full((batch_size, max_length), 0).long()
    for i, x in enumerate(batch):
        attention_mask[i, :len(x['input_ids'])] = torch.Tensor(x['attention_mask']).long()
    
    # do same for target_input_ids, but pad with pad_token
    max_target_len = max([len(x['target_input_ids']) for x in batch])
    target_input_ids = torch.full((batch_size, max_target_len), pad_token).long()
    for i, x in enumerate(batch):
        target_input_ids[i, :len(x['target_input_ids'])] = torch.Tensor(x['target_input_ids']).long()
    
    # do same for target_attention_mask, but pad with 0
    target_attention_mask = torch.full((batch_size, max_target_len), 0).long()
    for i, x in enumerate(batch):
        target_attention_mask[i, :len(x['target_input_ids'])] = torch.Tensor(x['target_attention_mask']).long()
    

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'target_input_ids': target_input_ids, 'target_attention_mask': target_attention_mask}

# train_dataset = dataset['train']
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collocator)
# optimizer over s_wte
optimizer = AdamW(s_wte.parameters(), lr=1e-3)
for batch in train_dataloader:
    # get batch
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['target_present'] = True
    embeds = s_wte(batch)
    # forward pass
    output = model(inputs_embeds=embeds['inputs_embeds'].to(device), attention_mask=embeds['attention_mask'].to(device))
    # get logits
    logits = output['logits']
    # do cross entropy loss
    logit_flat = logits.view(-1, logits.size(-1))
    labels_flat = embeds['labels'].view(-1).to(device)
    loss = loss_function(logit_flat, labels_flat)
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # get accuracy
    _, preds = torch.max(logit_flat, 1)
    # see if matches
    match = (preds == labels_flat).float()
    # only keep where labels_flat != -1
    match = match[labels_flat != -1]
    accuracy = match.mean()
    print(loss)
    print(accuracy)
    print(tokenizer.decode(preds[labels_flat != -1]))
    pass

# inputs = tokenizer(['Luke, may the force be with', 'Are tacos food?', 'My least favorite food is'], return_tensors="pt", padding=True)
# outputs = tokenizer(['you', 'yes', 'tuna'], return_tensors="pt", padding=True)
# inputs['target_input_ids'] = outputs['input_ids']
# inputs['target_attention_mask'] = outputs['attention_mask']
# inputs['target_present'] = True
# test = s_wte(inputs)
# breakpoint()
# output = model(inputs_embeds=test['inputs_embeds'], attention_mask=test['attention_mask'])
# logits = output['logits']
# # do cross entropy loss
# logit_flat = logits.view(-1, logits.size(-1))
# labels_flat = test['labels'].view(-1)
# loss = loss_function(logit_flat, labels_flat)
# 
# 
# # apply softmax to logits
# logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
# 
# batch_size = logprobs.size(0)
# # do cross entropy loss where target_mask
# loss = loss_function(logprobs, outputs['labels'])
# 
# pass
