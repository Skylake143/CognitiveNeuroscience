import os
import matplotlib.pyplot as plt
import models
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Based on the tutorial of https://medium.com/@myringoleMLGOD/simple-neural-network-for-dummies-in-pytorch-a-step-by-step-guide-38c4b1c914c0

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    if isinstance(loader.dataset, datasets.MNIST):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
    else: print("Checking accuracy on verification data")

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
    return accuracy

def train(train_loader, model, num_epochs=100):
    '''
    Trains the model 

    Based on the tutorial: https://medium.com/@myringoleMLGOD/simple-neural-network-for-dummies-in-pytorch-a-step-by-step-guide-38c4b1c914c0

    Parameters: 
        train_loader: DataLoader
            The DataLoader for the dataset to perform training on
        model: nn.Module
            The neural network model
    '''

    #Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Train the network
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()
    return model

def comparison_plot(accuracies):
    '''
    Creats a line plot and a bar plot.
    Compares the train, test and verification accuracy of the three models

    Parameters:
        accuracies: List
            List of dictionaries containing the accuracies 
            Each entry in the list is one model
    '''
    # Plot the accuracies
    models = [acc['model'] for acc in accuracies]
    train_accuracies = [acc['train_accuracy'] for acc in accuracies]
    test_accuracies = [acc['test_accuracy'] for acc in accuracies]
    verification_accuracies = [acc['verification_accuracy'] for acc in accuracies]

    plt.figure(figsize=(8, 6))
    plt.plot(models, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(models, test_accuracies, label='Test Accuracy', marker='o')
    plt.plot(models, verification_accuracies, label='Verification Accuracy', marker='o')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'SimpleNeuralNetwork/plots/lineplot{accuracies[0]['hyperparameters']}.png')
    plt.show()

    # Bar plot
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width - 0.05, train_accuracies, width, label='Train Accuracy')
    rects2 = ax.bar(x, test_accuracies, width, label='Test Accuracy')
    rects3 = ax.bar(x + width + 0.05, verification_accuracies, width, label='Verification Accuracy')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracies')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')

    # Function to add labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    #plt.grid(True)
    plt.savefig(f'SimpleNeuralNetwork/plots/barplot{accuracies[0]['hyperparameters']}.png')
    plt.show()

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
    num_epochs = 100

    #Load data
    train_dataset = datasets.MNIST(root="SimpleNeuralNetwork/dataset/", download=True, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root="SimpleNeuralNetwork/dataset/", download=True, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    verification_data_X = np.load("SimpleNeuralNetwork/dataset/verification/digits_X.npy")
    verification_data_y = np.load("SimpleNeuralNetwork/dataset/verification/digits_y.npy")

    # Convert to PyTorch tensors
    verification_data_X = torch.tensor(verification_data_X, dtype=torch.float32)
    verification_data_y = torch.tensor(verification_data_y, dtype=torch.float32)

    # Create TensorDataset and DataLoader
    verification_dataset = TensorDataset(verification_data_X, verification_data_y)
    verification_loader = DataLoader(verification_dataset, batch_size=batch_size, shuffle=False)

    #Train models
    for ModelType in [models.NNOneHiddenLayer, models.NNThreeHiddenLayers, models.NNConvolutionalThreeHiddenLayers]:
        model = ModelType(28,28,num_classes=num_classes).to(device)
        model_name = model.__class__.__name__
        model_path = f"SimpleNeuralNetwork/models/{model_name}lr{learning_rate}batch{batch_size}epochs{num_epochs}.pth"
        if not os.path.exists(model_path):
            print(f"Training model: {model_name}")
            output_model = train(train_loader, model, num_epochs)
            
            torch.save(output_model.state_dict(), model_path)
            print(f"Saved model: {model_name}")
        else: 
            print(f"Model {model_name} already trained")

    # List to store accuracies for plotting
    accuracies = []

    #Load models and calculate final accuracy training and test sets
    for ModelType in [models.NNOneHiddenLayer, models.NNThreeHiddenLayers, models.NNConvolutionalThreeHiddenLayers]:
        model = ModelType(28, 28, num_classes=num_classes).to(device)
        model_name = model.__class__.__name__
        model_path = f"SimpleNeuralNetwork/models/{model_name}lr{learning_rate}batch{batch_size}epochs{num_epochs}.pth"

        if os.path.exists(model_path):
            print(f"Loaded model: {model_name} from {model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=True))

            accuracy_train = check_accuracy(train_loader, model)
            accuracy_test = check_accuracy(test_loader, model)
            accuracy_verification = check_accuracy(verification_loader, model)

            accuracies.append({
                'model': model_name,
                'hyperparameters': f'lr{learning_rate}batch{batch_size}epochs{num_epochs}',
                'train_accuracy': accuracy_train,
                'test_accuracy': accuracy_test,
                'verification_accuracy': accuracy_verification
            })
        else:
            print(f"Model {model_name} not found at {model_path}")

    comparison_plot(accuracies)