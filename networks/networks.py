import torch.nn as nn
import torch.nn.functional as F

from . import utils



class MLP(utils.ReparamModule):
    supported_dims = {9, 28}  # Add 9 as the supported dimension for your input images
    def __init__(self, state):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * state.input_size * state.input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, state.num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


