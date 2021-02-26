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

        self.efficient_net = EfficientNet.from_pretrained(arch)
        # print(EfficientNet.get_image_size('efficientnet-b4'))
        self.flatten_size = flatten_size
        self.fc1 = nn.Linear(self.flatten_size,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,output_dim)
        
    def forward(self, x):   
        # with torch.no_grad():
        # x = self.resnet(x)
        # print("before "+str(x.shape))
        with torch.no_grad():
            x = self.efficient_net.extract_features(x)
        # print(f'before squeeze {x.shape}')
        x = torch.reshape(x,[-1, self.flatten_size])
        # print('After reshaping',x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
