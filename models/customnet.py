import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.flatten = nn.Flatten(2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=4)

        self.conv2 = nn.Conv2d(64,128, kernel_size=3, padding=1, stride=2)

        self.conv3 = nn.Conv2d(128,256, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(256,512, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(512,1024, kernel_size=3, padding=1, stride=2)

        self.fc1 = nn.Linear(1024, 200)

        # Add more layers...
        # self.linear_relu_stack= nn.Sequential( #ongi strriude diviso 2
        #     nn.Linear(14*14*512 , 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024 , 512),
        #     nn.ReLU(),
        #     nn.Linear(512 , 256),
        #     nn.ReLU(),
        #      nn.Linear(256 , 200)
        #    )
        # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        x = self.conv1(x).relu()

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        x=self.conv5(x)

        #x = self.batch_norm4(x).relu()

        x = self.flatten(x).mean(-1)

        return self.fc1(x)