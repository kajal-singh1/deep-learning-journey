import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# ─── Datasets & Dataloaders ───────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = ImageFolder(root="./data/train", transform=transform)
testset  = ImageFolder(root="./data/test",  transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset,  batch_size=64)

# ─── Build the CNN ────────────────────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 64x64 → 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 64x64 → 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32 → 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # 32x32 → 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16 → 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                             # 16x16 → 8x8
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)                             # binary → 1 output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


model     = CNN()
criterion = nn.BCEWithLogitsLoss()                        # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── Training + Validation ────────────────────────────────────────────────────

epochs = 10

for epoch in range(epochs):

    # Training
    model.train()
    epoch_training_loss = 0.0

    for images, labels in trainloader:
        labels = labels.float().unsqueeze(1)              # shape: [batch, 1]

        optimizer.zero_grad()
        output = model.forward(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_training_loss += loss.item()

    epoch_training_loss = epoch_training_loss / len(trainloader)

    # Validation
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            labels  = labels.float().unsqueeze(1)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_val_loss += loss.item()

    running_val_loss = running_val_loss / len(testloader)

    print(f"epoch={epoch+1}/{epochs} | Train Loss: {epoch_training_loss:.4f} | Val Loss: {running_val_loss:.4f}")

# ─── Evaluate CNN ─────────────────────────────────────────────────────────────

correct_labels = 0
total_labels   = 0

model.eval()
with torch.no_grad():
    for images, labels in testloader:
        outputs   = model.forward(images)
        predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze(1)

        correct_labels += (predicted == labels).sum().item()
        total_labels   += labels.size(0)

print(f"accuracy = {correct_labels / total_labels * 100:.2f}%")