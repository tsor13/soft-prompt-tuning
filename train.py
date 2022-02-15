# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from datasets import load_dataset
from lm_utils import get_device_map
from pdb import set_trace as breakpoint
from tqdm import tqdm

model_name = 'gpt2'
# model_name = 'gpt2-xl'
# model_name = 'EleutherAI/gpt-j-6B'

device = 'cuda'
# device = 'cpu'

# load imdb
dataset = load_dataset('imdb')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
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

train_dataset = dataset['train'].map(preprocess_function, batched=False)
train_dataset = train_dataset.map(label_function, batched=False)


from SoftPromptModel import SoftPromptModel

# model
n_tokens = 10
initialize_from_vocab = True

model = AutoModelForCausalLM.from_pretrained(model_name)
print(model_name)

# get the number of attention layers
n_blocks = model.config.n_layer
if torch.cuda.is_available():
    # get all available GPUs
    gpus = np.arange(torch.cuda.device_count())
    device = 'cuda:0'
    if len(gpus) > 1:
        device_map = get_device_map(gpus, n_blocks)
        model.parallelize(device_map)
    else:
        model = model.to(device)
    print(f'Loaded model on {len(gpus)} GPUs.')
else:
    device = 'cpu'
    print('Loaded model on cpu.')

soft_prompt_model = SoftPromptModel(
    bm=model,
    n_append=n_tokens,
    n_prepend=n_tokens,
)


# loss function
loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)

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
# optimizer = AdamW(s_wte.parameters(), lr=1e-4)
# optimizer = AdamW(s_wte.parameters(), lr=1e-3)
optimizer = AdamW(soft_prompt_model.parameters(), lr=1e-3)

indices_of_interest = [tokenizer.encode(label)[0] for label in ['positive', 'negative']]

max_batches = 300

for i, batch in tqdm(enumerate(train_dataloader), total=max_batches):
    # get batch
    batch = {k: v.to(device) for k, v in batch.items()}
    # send through soft prompt model
    output = soft_prompt_model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        target_ids=batch['target_input_ids'],
        target_mask=batch['target_attention_mask'],
    )
    loss = output[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.item()}')

    if i >= max_batches:
        break
    # # get logits
    # logits = output['logits']

    # # CROSS ENTROPY OF ANY LABEL
    # # get logprobs
    # logprobs = F.log_softmax(logits, dim=-1)
    # label_logprobs = logprobs[:, :, indices_of_interest]
    # # do softmax
    # split_locations = output['target_index']
    # # only keep logits at [0, split_locations[0]], ..., [n, split_locations[n]]
    # inds1 = np.arange(0, split_locations.shape[0])
    # inds2 = split_locations.flatten()
    # # NEED TO SUBTRACT ONE TO FIX OFF BY ONE ERROR
    # label_logprobs = label_logprobs[inds1, inds2-1]
    # # maximize weight on any of the labels
    # # get probs and calculate cross entropy
    # probs = torch.exp(label_logprobs)
    # all_probs = torch.sum(probs, dim=1)
    # # get loss
    # any_ce_loss = -(torch.log(all_probs)).mean()
    # print(f'Any CE loss: {any_ce_loss.item()}')

    # # MUTUAL INFORMATION BETWEEN LABELS
    # # normalize probs so that they sum to 1
    # probs = probs / torch.sum(probs, dim=1).unsqueeze(1)
    # # TODO - make more numerically stable??
    # cond_entropy = -(probs * torch.log(probs)).sum(dim=1)
    # overall_dist = torch.mean(probs, dim=0)
    # marginal_entropy = -(overall_dist * torch.log(overall_dist)).sum()
    # mutual_info = marginal_entropy - cond_entropy.mean()
    # mi_loss = -mutual_info
    # print(f'Mutual info loss: {mi_loss.item()}')
    
    # # CROSS ENTROPY
    # # do cross entropy loss
    # logit_flat = logits.view(-1, logits.size(-1))
    # # labels_flat = batch['labels'].view(-1).to(device)
    # labels_flat = output['labels'].view(-1).to(device)
    # ce_loss = loss_function(logit_flat, labels_flat)
    # print(f'Cross entropy loss: {ce_loss.item()}')
    # # backprop
    # # optimizer.zero_grad()
    # # loss.backward()
    # # optimizer.step()
    # # get accuracy
    # _, preds = torch.max(logit_flat, 1)
    # # see if matches
    # match = (preds == labels_flat).float()
    # # only keep where labels_flat != -100
    # match = match[labels_flat != -100]
    # accuracy = match.mean()
    # # sanity check
    # # print(torch.argmax(logits[inds1, inds2-1], 1) == preds[labels_flat != -1])
    # print(f'Accuracy: {accuracy.item()}')
    # print(tokenizer.decode(preds[labels_flat != -100]))

    # # loss = any_ce_loss + mi_loss + ce_loss
    # # loss = mi_loss + any_ce_loss
    # # loss = mi_loss + ce_loss
    # # print(f'Total loss: {loss.item()}')
    # # optimizer.zero_grad()
    # # loss.backward()
    # # optimizer.step()

breakpoint()
test = 'This was the worst movie I\'ve ever seen. Leonardo Dicaprio, more like Leonardo DeCRAPio. The writing was okay, but the acting was terrible. I am never going to see a movie like this again.'
# tokenize
example = tokenizer(test, return_tensors='pt')
output = soft_prompt_model( input_ids=example['input_ids'].to(device), attention_mask=example['attention_mask'].to(device),)
print(tokenizer.decode(output['logits'][0][-1].argmax()))
breakpoint()
# TODO
# Seems to mostly work. Biggest questions: how to autoregressively sample?
# make sure the alternate losses and accuracies are all lined up
pass