import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784
hidden_size = 128
output_size = 10
lr = 0.01
epochs = 30
batch_size = 64

# Functions
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_prime(x): # derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

def cross_entropy(outputs, labels):
    return -torch.sum(labels * torch.log(outputs))

# Load Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# Weight and bias initialization
W1 = torch.randn(input_size, hidden_size) * 0.01
b1 = torch.zeros(hidden_size)

W2 = torch.randn(hidden_size, output_size) * 0.01
b2 = torch.zeros(output_size)

train_losses = []
test_losses = []

for epoch in range(1, epochs+1):
    for images, labels in train_loader:
        # Flatten the images
        images = images.view(-1, 784)

        # One-hot encoding of labels
        labels_onehot = torch.zeros(len(labels), 10)
        labels_onehot[torch.arange(len(labels)), labels] = 1
        
        # --- FORWARD PASS ---
        z1 = torch.mm(images, W1) + b1
        a1 = sigmoid(z1)
        z2 = torch.mm(a1, W2) + b2
        outputs = softmax(z2)
        
        loss = cross_entropy(outputs, labels_onehot) / len(labels)
        
        # --- BACKWARD PASS ---
        dz2 = outputs - labels_onehot
        dW2 = torch.mm(a1.t(), dz2)
        db2 = torch.sum(dz2, dim=0)

        da1 = torch.mm(dz2, W2.t())
        dz1 = da1 * sigmoid_prime(z1)
        dW1 = torch.mm(images.t(), dz1)
        db1 = torch.sum(dz1, dim=0)

        # Update weights and biases
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Evaluate on test set
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784)
            z1 = torch.mm(images, W1) + b1
            a1 = sigmoid(z1)
            z2 = torch.mm(a1, W2) + b2
            outputs = softmax(z2)
            
            labels_onehot = torch.zeros(len(labels), 10)
            labels_onehot[torch.arange(len(labels)), labels] = 1

            test_loss += cross_entropy(outputs, labels_onehot) / len(labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    train_losses.append(loss)

    print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Test Loss: {test_loss/len(test_loader):.4f} - Test Accuracy: {correct/len(test_loader.dataset):.4f}")

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
