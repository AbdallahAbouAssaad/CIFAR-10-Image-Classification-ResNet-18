import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # flip images randomly
    transforms.RandomCrop(32, padding=4), # random crop with padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),               # convert to tensor [0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)), # CIFAR-10 mean/std (already calculated)
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)) # Must be put at the end because Images dont have a shape
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Downloading the dataset and applying the preprocessing pipeline

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,  # 128 images per step
    shuffle=True,   # ensures batches are randomized each step
    num_workers=2)  # parallel loading

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # roughly un-normalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:16], nrow=4))
print(' '.join(f'{classes[labels[j]]}' for j in range(16)))

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load pretrained ResNet18
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)  # most up-to-date weights

# Modify the final fully connected layer for CIFAR-10
num_ftrs = resnet.fc.in_features  # original input features
resnet.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

# Move model to GPU if available
resnet = resnet.to(device)

# Freeze everything first
for param in resnet.parameters():
    param.requires_grad = False

# Unfreeze last 3 ResNet blocks (layer2, layer3, layer4) + final FC
for name, param in resnet.named_parameters():
    if "layer2" in name or "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # multiclass classification

# Different LR for different layers
optimizer = optim.Adam([
    {"params": resnet.layer2.parameters(), "lr": 0.0001},
    {"params": resnet.layer3.parameters(), "lr": 0.0005},
    {"params": resnet.layer4.parameters(), "lr": 0.001},
    {"params": resnet.fc.parameters(), "lr": 0.001}
])

# Get a batch of training data
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Forward pass
outputs = resnet(images)
print("Output shape:", outputs.shape)  # should be [batch_size, 10]

num_epochs = 100
patience = 5
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients (important!)
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()
        optimizer.step()

        # Track stats
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}")
            running_loss = 0.0

    # Early stopping check at the end of each epoch
    epoch_loss = running_loss / len(trainloader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0  # reset counter
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Stopping early at epoch {epoch+1}, no improvement for {patience} epochs.")
        break

print("Finished Training ✅")

correct = 0
total = 0
resnet.eval()  # set to evaluation mode
with torch.no_grad():  # no gradient tracking needed
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test images: {100 * correct / total:.2f}%")
