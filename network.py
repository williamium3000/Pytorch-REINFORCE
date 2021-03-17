import torch
from torch import nn
import random
import numpy as np
import torchvision
import torchvision.utils
import torch
import torch.nn as nn
from torchvision import models
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class ravel(nn.Module):
    def __init__(self):
        super(ravel, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)
class PGnetwork(nn.Module):
    def __init__(self, num_obs, num_act):
        super(PGnetwork, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(num_obs, 64, True),
            nn.LeakyReLU(),
            # nn.Linear(64, 64, True),
            # nn.LeakyReLU(),
            nn.Linear(64, num_act, True),
            nn.Softmax(1)

        )


    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ == "__main__":
    pass
