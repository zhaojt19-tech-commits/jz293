from MyDataset_10K import *
import torch.nn.functional as F
from torch import nn

if __name__ == '__main__':
    # input = torch.randn(2, 6, 1, 4, 3, requires_grad=True)
    # print(input)
    # target = torch.randint(6, (2, 1, 4, 3), dtype=torch.int64)
    # print(target)
    # loss = F.cross_entropy(input, target)
    # print(loss)
    # loss.backward()

    embedding = nn.Embedding(10, 2)  # 10个词，每个词用2维词向量表示
    print(embedding.weight)