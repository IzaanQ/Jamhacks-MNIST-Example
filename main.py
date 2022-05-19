import torch
import torch.nn as nn
from torchvision import datasets, transforms

def train(model, train_loader, loss_function, optimizer):
    model.train()

    total_loss = 0

    for images, labels in train_loader:
        batch_size = images.shape[0]
        images = images.view(batch_size, 784)

        optimizer.zero_grad()

        output = model(images)
        loss = loss_function(output, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)
    return avg_loss


def main(N_EPOCHS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_data = datasets.MNIST("./data", download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64)

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train(model, train_loader, loss_function, optimizer)

        print("Epoch: " + str(epoch))
        print("Loss: " + str(train_loss))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "mnist_nn.pt")
            print("Saved model to mnist_nn.pt!")
        
        print()

main(20)
