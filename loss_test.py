import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 畳み込み層1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 畳み込み層2
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全結合層1
        self.fc2 = nn.Linear(128, 10)  # 全結合層2（出力層）

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.dropout(x, 0.25)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, 0.5)
        x = self.fc2(x)
        return x
class AdaCosLoss(nn.Module):
    def __init__(self, num_classes=10):
        super(AdaCosLoss, self).__init__()
        self.num_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)

    def forward(self, logits, labels):
        cos_theta = F.cosine_similarity(logits, F.one_hot(labels, num_classes=self.num_classes).float(), dim=-1)
        return F.cross_entropy(self.s * cos_theta, labels)

class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        cos_theta = F.cosine_similarity(logits, F.one_hot(labels, num_classes=10).float(), dim=-1)
        phi = cos_theta - self.m
        return F.cross_entropy(self.s * phi, labels)
def train_and_evaluate(model, loss_fn, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {epoch}: Test accuracy: {100. * correct / len(test_loader.dataset)}%")

# データローダーの設定
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transforms.ToTensor()), batch_size=64)

# AdaCosでの訓練と評価
print("Training with AdaCos")
model = SimpleCNN()
train_and_evaluate(model, AdaCosLoss(), train_loader, test_loader)

# ArcFaceでの訓練と評価
print("Training with ArcFace")
model = SimpleCNN()
train_and_evaluate(model, ArcFaceLoss(), train_loader, test_loader)
