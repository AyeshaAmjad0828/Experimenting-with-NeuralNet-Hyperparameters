import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

# Loading CIFAR10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Set up data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# training function
def train(model, criterion, optimizer, train_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # L1 and L2 regularization
            l1_lambda = 0.001
            l2_lambda = 0.001
            l1_reg = torch.tensor(0., requires_grad=True)
            l2_reg = torch.tensor(0., requires_grad=True)

            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
                l2_reg = l2_reg + torch.norm(param, 2)

            loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%")

# Hyperparameters
num_classes = 10
learning_rate = 0.001
epochs = 10

# VGG model
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, num_classes)  

# Different optimizers
optimizers = [
    optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9),
    optim.Adam(model.parameters(), lr=learning_rate),
]

for optimizer in optimizers:
    criterion = nn.CrossEntropyLoss()
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = optimizer

    print(f"\nTraining with optimizer: {optimizer.__class__.__name__}")
    train(model, criterion, optimizer, train_loader, epochs)
