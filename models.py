## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## Input is a image with 224x224 pixels
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional Layer 1: in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1 (default)
        # Output Image: 220x220 after pooling 110x110 (（224-5+0）/1)+1, so how we have 110 as image size/map
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # Convolutional Layer 2: in_channels = 32, out_channels = 64, kernel_size = 4, stride = 1 (default)
        # Output Image: 106x106 after pooling 53x53, repeat the process again while geting double of the output features
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Convolutional Layer 3: in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1 (default)
        # Output Image: 51x51 after pooling 25x25, again double the ouput features, however start to taking smaller kernels to look details
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Convolutional Layer 4: in_channels = 128, out_channels = 256, kernel_size = 2, stride = 1 (default)
        # Output Image: 24x24 after pooling 12x12, double the ouput to get 256 intotal, keep using smaller kernal so to check details
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # start add dropout within the Convolutional layers to avoid overfitting, add each drop after each conv2d layer just before relu
        # keep the number in convolutional layer small so as to avoid missing/lossing important features
       # self.dropout1 = nn.Dropout(p=0.1)
       # self.dropout2 = nn.Dropout(p=0.1)
      # self.dropout3 = nn.Dropout(p=0.1)
       # self.dropout4 = nn.Dropout(p=0.1)
        
        # two extra dropout to fit the process from 1000 features to 136 points
        #self.dropout5 = nn.Dropout(p=0.4)
        #self.dropout6 = nn.Dropout(p=0.4)
        # finally, create the full connected output layer, each image is size of 6*6 with total of 512 of it, the output is 1024 features
        self.fc1 = nn.Linear(512*6*6, 1024)
        
        # two extra linear connected layer after each extra drop off as mentioned above;
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)
        
          # Dropout
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        ## add a dropout layer after each of the cnn layer,the order is Convolution >> Activation >> Pooling >> Dropout        
     
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between, using the dropout layer only in the full connected layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
