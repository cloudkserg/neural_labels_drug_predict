import torch

n_embd = 10
n_hidden = 64
vocab_size = 27
block_size = 3

g = torch.Generator().manual.seed(1)
C = torch.randn((vocab_size, n_embd), generator = g)

W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.1

W2 = torch.randn((n_hidden, vocab_size), generator=g)*0.1
b2 = torch.randn(vocab_size, generator=g)*0.1

bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
	p.requires_grad = True

# forward pass
emb = C[X]
embcat = emb.view(emb.shape[0], -1)

hprebn = embcat @ W1 + b1

bnmean1 = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmean1
bndiff2 = bndiff ** 2

bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
# diff = (X-U) var = (X-U)^2 inv = sum(var)/(n-1))^1/2
# raw = (X-U)/inv
bnraw = bndiff * bnvar_inv
# G*(X-U)/@ +B 
hpreact = bngain*bnraw + bnbias

h = torch.tanh(hpreact)

logits = h @ W2 + b2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes

counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv

# norm_logits = logit-max() 
# probs = e^logit_norm/sum(e^logit_norm)


import torch


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

xb, yb = get_batch('train')
logits, loss = m(xb, yb)
optmizer.zero_grad(set_to_none=True)
loss.backwad()
optmizer.step()

decode(
	m.generate(idx=torch.zeros((1,1), dtype=torch.long)
		, max_new_tokens=100)[0].to_list()
)


device = 'cuda' if torch.cuda.is_available else 'cpu'

x.to(device)
y.to(device)

model = BigramLanguageModel
model.to(device)


import torch.nn as nn

class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()






logprobs = probs.log()
loss = - logprobs(range(n), Yb).mean()

x = torch.randn(2,2,3)

xbow = torch.zeros((2,2,3))
for b in range(2):
	for t in range(2):
		xt = x[b, :t+1]
		xbow[t] = torch.mean(xt, 0)

tril = torch.tril(torch.ones(3,3))
wei = torch.zeros(3,3)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
xbow = wei @ x


C = 24
head_size = 16
key = nn.Linear((C, head_size), bias=False)
query = nn.Linear((C, head_size), bias=False)
value = nn.Linear((C, head_size), bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(3,3))
wei = wei.masked_fil(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
out = wei @ value

self.register_buffer('tril', torch.tril(torch.ones(3,3)))

