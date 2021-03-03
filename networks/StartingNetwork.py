import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, input_dim, output_dim):
        super(StartingNetwork, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        self.conv1 = nn.Conv2d(input_dim, 3, kernel_size = 3,padding=1)      
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
        # print(EfficientNet.get_image_size('efficientnet-b4'))

        # self.efficient_net = nn.Sequential(*list(self.efficient_net.children())[:-1])
        # self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        # # self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(256, 10, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_size = 2048
        # 512
        # 2560*12*12
        # 1000
        # 10 * 35 *48
        # 33600
        # 10*50*37
        self.fc1 = nn.Linear(self.flatten_size,1024)
        # self.fc2 = nn.Linear(1024,256)
        self.fc2 = nn.Linear(1024,output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        # with torch.no_grad():
        x = self.resnet(x)
        # print("before "+str(x.shape))
        # with torch.no_grad():
        # x = self.efficient_net.extract_features(x)
        # print(f'before squeeze {x.shape}')
        x = torch.reshape(x,[-1, self.flatten_size])
        # print('After reshaping',x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        return x
