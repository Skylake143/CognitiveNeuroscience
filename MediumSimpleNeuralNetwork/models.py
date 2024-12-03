import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class NNOneHiddenLayer(nn.Module):
    def __init__(self, input_height, input_width,num_classes):
        """
        Basic fully connected feedforward neural network with
            input layer: flattens from 4D tensor to 2D tensor
            hidden layer: Fully connected layer with 50 neurons
            output layer: fully connected layer with num_class outputs (10  for MNIST (digits 0 to 9))

        Parameters:
            input_size: int
                The size of the input
            num_classes: int
                The number of classes we want to predict
        """
        super(NNOneHiddenLayer, self).__init__() #Refer to parent class of NNOneHiddenLayer (nn.module) and call init
        self.flatten = nn.Flatten() #Input layer: flattens 2D image to 1D data
        self.fc1 = nn.Linear(input_height*input_width, 128)
        self.fcoutput= nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x=self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fcoutput(x)
        return x
    
class NNThreeHiddenLayers(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        """
        Basic fully connected feedforward neural network with
            input layer: for MNIST 784 (28x28)
            hidden layer: 3 layers of 50 neurons
            output layer: for MNIST 10 (digits 0 to 9)

        Parameters:
            input_size: int
                The size of the input
            num_classes: int
                The number of classes we want to predict
        """
        super(NNThreeHiddenLayers, self).__init__()
        self.flatten = nn.Flatten() #Flatten 
        self.fc1 = nn.Linear(input_height*input_width, 512) #First hidden layer should be multiple of input size
        self.fc2 = nn.Linear(512, 256)# Reduction of nodes in hidden layer 2 by half
        self.fc3 = nn.Linear(256, 128) # Again reduction by half
        self.fcoutput = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x= self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fcoutput(x)
        return x

    
class NNConvolutionalOneHiddenLayer(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        """
        Basic convolutional neural network with a convultional connection and a linear connection
            input layer: 1 for loading the whole image
            hidden layer: 50
            output layer: for MNIST 10 (digits 0 to 9)

        Parameters:
            input_size: int
                The size of the input
            num_classes: int
                The number of classes we want to predict
        """
        super(NNConvolutionalOneHiddenLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels =1, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU()) # Convolutional layer with 32 filters
        self.fc1 = nn.Linear(32 * input_height * input_width, 128)
        self.fcoutput = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.fcoutput(x)
        return x
    
class NNConvolutionalThreeHiddenLayers(nn.Module):
    def __init__(self,input_height, input_width, num_classes):
        super(NNConvolutionalThreeHiddenLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Calculate the size after conv and pooling layers
        conv_output_height = (input_height - 4) // 2
        conv_output_width = (input_width - 4) // 2
        self.fc1 = nn.Linear(64 * conv_output_height * conv_output_width, 128)
        self.fcoutput = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fcoutput(x)
        output = F.log_softmax(x, dim=1)
        return output