import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F



batch_size = 32
block_size = 8
max_iter = 5000
learning_rate = 1e-3
n_embed = 32
n_head = 4
n_layers = 8
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



class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,head_size):
        super().__init__()
        self.h = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embed,n_embed)

    def forward(self,x):
        out =  torch.cat([h(x) for h in self.h], dim=-1)
        return self.proj(out)

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
      
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        out = wei @ v
        return out
class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed,n_embed),nn.ReLU(),nn.Linear(n_embed,n_embed))
    def forward(self,x):
        return self.net(x);

class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.mul = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self,x):
        x = x+self.mul(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class BiagramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding = nn.Embedding(block_size,n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layers)])                            )
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        

    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_embed = self.token_embedding_table(idx) #(B,T,n_embed)
        pos_embed = self.position_embedding(torch.arange(T,device=device)) #(T,n_embed)
        x = tok_embed+pos_embed #(B,T,n_embed)+(T,n_embed)-->(B,T,n_embed)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        if(targets is None):
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

m = BiagramLanguageModel()
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