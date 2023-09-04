import torch.nn as nn
import torch

embedding = nn.Embedding(3,8,2)
print(embedding(torch.tensor(0)))
