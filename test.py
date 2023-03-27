words = open('names.txt', 'r').read().splitlines()
# dict = {}
# for w in words:
#     chs = ['<s>'] + list(w) + ['<e>']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         tup = (ch1, ch2)
#         dict[tup] = dict.get(tup, 0) + 1

# items = sorted(dict.items(), key=lambda kv: -kv[1])
# print(len(items))
# print(list(items)[0])


import torch
import matplotlib.pyplot as plt

N = torch.zeros((28, 28), dtype=torch.int32)


chars = sorted(set(''.join(words)))
stoi = {c:i for i, c in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = (stoi[ch1], stoi[ch2])
        N[ix1, ix2] += 1

# plt.imshow(N)

# plt.show()
p = N[0].float()
p = p / p.sum()
print(p)
