import torch
import torch.nn as nn
from torchvision import datasets, transforms

# NOTE
# Read through the setup code in function main() to 
# get an idea of what's going on in function train()

# Training functions
# Steps the model for each batch of input,label pairs in the data loader
def train(model, train_loader, loss_function, optimizer):
    # Gotta set the model to training mode
    # (Doesn't do anything for us but will for more complex neural nets)
    model.train()

    total_loss = 0 # Used to calculate average loss

    # Iterate through each input,label pair in the data loader
    for images, labels in train_loader:
        # images are tensors of dimensions (batch_size, 28, 28)

        batch_size = images.shape[0]
        images = images.view(batch_size, 784) # Flatten the each image in the batch

        # Resets the optimizer
        # grad stands for gradient
        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Loss calculation
        loss = loss_function(output, labels)

        # Backward pass (backpropagation)
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Add loss to total loss
        total_loss += loss.item()

    # Calculate and return average loss
    avg_loss = total_loss/len(train_loader)
    return avg_loss


# Definition of Epoch: One epoch of training occurs after the model is trained
#                      once on every batch of inputs in the data loader.

# What is a Tensor?
#   A tensor is a generalization of vectors and matrices (1d and 2d arrays of numbers, respectively)
#   A vector would be a 1d tensor, a matrix is a 2d tensor. Adding a third axis, we get a 3d tensor.

def main(N_EPOCHS):
    ##############
    # SETUP CODE #
    ##############

    # transform object
    # Applies some transformations to the input data
    transform = transforms.Compose([
        transforms.ToTensor(),              # Converts image into a tensor
        transforms.Normalize((0.5), (0.5))  # Normalizes values to roughly within the range 0 - 1
    ])

    # Get training data
    train_data = datasets.MNIST("./data", download=True, train=True, transform=transform)

    # Load data onto a data loader
    # Data loaders do a lot of the heavy lifting for us when it comes to data preparation
    # They handle bundling together inputs with labels and can be iterated over with a for loop.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)

    # This is our neural network
    # nn.Sequential is a sequence of operations that define our model
    # In more serious applications, you wouldn't use nn.Sequential.
    model = nn.Sequential(
        nn.Linear(784, 128),    # A Linear layer with 784 inputs and 128 outputs
                                # Another note: nn layers and activation functions are separate
                                # You perform the weighted sum first then apply activation function

        nn.ReLU(),              # This is our activation function
                                # ReLU is a bit better than sigmoid, sorry sigmoid :( 
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    # Our loss function
    loss_function = nn.CrossEntropyLoss()

    # SGD stands for "Stochastic Gradient Descent"
    # Don't worry about what stochastic means
    # SGD is an approximation of actual gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) # lr is the learning rate
                                                               # (ie how much the optimizer nudges each weight value)

    # The main training loop
    for epoch in range(1, N_EPOCHS + 1):
        # Apply an optimizer step once for each input,label pair in train_loader
        # And get the average loss.
        train_loss = train(model, train_loader, loss_function, optimizer)

        # Logs
        print("Epoch: " + str(epoch))
        print("Loss: " + str(train_loss))

        # For every fifth epoch, save the model to mnist_nn.pt
        if epoch % 5 == 0:
            # PyTorch models store their weight values in a "state dict"
            # so torch.save just writes that to a file.
            torch.save(model.state_dict(), "mnist_nn.pt")
            print("Saved model to mnist_nn.pt!")
        
        print()

# We train for about 20 epochs
main(20)
