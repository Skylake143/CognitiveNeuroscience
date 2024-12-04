import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models

def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            #x = x.reshape(x.shape[0], -1)

            # Forward pass: compute the model output
            scores = model(x)

            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  # Set the model back to training mode

def train(train_loader, model):
    '''
    Trains the model 

    Parameters: 
        train_loader: DataLoader
            The DataLoader for the dataset to perform training on
        model: nn.Module
            The neural network model
    '''
    #Hyperparameters
    num_epochs = 100

    #Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Train the network
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            
            # Reshape data to (batch_size, input_size), if it is directly fed into linear network
            #data = data.reshape(data.shape[0], -1)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

if __name__ =="__main__":
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #Define hyperparameters
    input_height = 28  # Image height
    input_width = 28 # Image width
    num_classes = 10  # digits 0-9
    learning_rate = 0.001
    batch_size = 64

    #Load data
    train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the network
    model = models.NNConvolutionalOneHiddenLayer(28,28, num_classes=num_classes).to(device)

    #Train model
    train(train_loader, model)
    
    #Final accuracy check on training and test sets
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)