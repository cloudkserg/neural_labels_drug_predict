import string
import torch
import matplotlib.pyplot as plt

chars = string.ascii_lowercase
itos = {i:char for i, char in enumerate(chars)}
C = torch.randn((26, 2))

# plt.figure(figsize=(8,8))
# plt.scatter(C[:, 0].data, C[:, 1].data, s= 200)
# for i in range(C.shape[0]):
# 	plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], 
# 		ha='center', va='center', color='white')
# plt.grid('minor')

# plt.show()

plt.figure(figsize=(8,8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
	plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
		va='center', ha='center', color='white'
	)
plt.grid('minor')
plt.show()