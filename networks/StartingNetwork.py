import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim, arch, flatten_size):
        super(StartingNetwork, self).__init__()
        # self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.conv1 = nn.Conv2d(input_dim,3,kernel_size=5,padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=1)
        self.efficient_net = EfficientNet.from_pretrained(arch)
        # print(EfficientNet.get_image_size('efficientnet-b4'))
        self.flatten_size = flatten_size
        self.fc1 = nn.Linear(self.flatten_size,128)
        self.fc2 = nn.Linear(128,output_dim)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):   
        # with torch.no_grad():
        # x = self.resnet(x)
        # print("before "+str(x.shape))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # with torch.no_grad():
        x = self.efficient_net.extract_features(x)
        # print(f'before squeeze {x.shape}')
        x = torch.reshape(x,[-1, self.flatten_size])
        # print('After reshaping',x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
