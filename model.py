import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F



batch_size = 32
block_size = 8
max_iter = 3000
learning_rate = 1e-3
n_embed = 32
n_head = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()


print("Length of dataset in charachters:",len(text))

chars = sorted(list(set(text)))

vocab_size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[x] for x in l])


data = torch.tensor(encode(text),dtype=torch.long)
print(data.shape)



n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)


def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # 4x8 tensors  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device=device)
    y = y.to(device=device)
    return x,y

xb, yb = get_batch('train')



class BiagramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding = nn.Embedding(block_size,n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_embed = self.token_embedding_table(idx) #(B,T,n_embed)
        pos_embed = self.position_embedding(torch.arange(T,device=device)) #(T,n_embed)
        x = tok_embed+pos_embed #(B,T,n_embed)+(T,n_embed)-->(B,T,n_embed)
        logits = self.lm_head(x)
        if(targets==None):
            loss = None
        else:
            
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):

        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

m = BiagramLanguageModel(vocab_size)
m.to(device=device)


optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)


for step in range(1000):
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=1000)[0].tolist()))