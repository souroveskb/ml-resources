import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import  random_split, DataLoader

import pytorch_lightning as pl
import torchmetrics


# entrypoints = torch.hub.list('pytorch/vision', force_reload=False)

# for e in entrypoints:
#     if "resnet" in e:
#         print(e)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet(pl.LightningModule):
    pass


# set device if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyper parameters
input_size = 28 * 28 # 784
num_classes = 10 # 0-9
learning_rate = 1e-3
batch_size = 64
num_epochs = 3

# Load Dataset
full_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

# split train val
train_data, val_data = random_split(full_dataset, [55000, 5000])

