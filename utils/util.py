import torch
import torch.nn as nn

if __name__ == '__main__':
    softmax = nn.Softmax(dim=-1)
    score = torch.full((2, 2), float("-inf"))
    score[0][0] = 1
    res = softmax(score)
    print(res)