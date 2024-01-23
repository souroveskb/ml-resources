import torch
from torch import nn

entrypoints = torch.hub.list('pytorch/vision', force_reload=False)

for e in entrypoints:
    if "resnet" in e:
        print(e)