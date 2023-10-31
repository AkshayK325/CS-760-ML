import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784
hidden_size = 128
output_size = 10
lr = 0.01
epochs = 30
batch_size = 64

# Neural Network definition
class SimpleNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,init_method):
        super(SimpleNN, self).__init__()
        self.input_size=input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Weight Initialization
        if init_method == "zeros":
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc2.weight)
        elif init_method == "random":
            nn.init.uniform_(self.fc1.weight, -1, 1)
            nn.init.uniform_(self.fc2.weight, -1, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Testing function
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# Initialize model, optimizer, and criterion
model = SimpleNN(input_size,hidden_size,output_size,init_method="any")
model = SimpleNN(input_size, hidden_size, output_size, init_method="zeros")
model = SimpleNN(input_size, hidden_size, output_size, init_method="random")

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss, test_accuracy = test(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

# Plotting learning curve
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
