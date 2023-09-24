import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ハイパーパラメータの設定
epoch = 100
arcface_s = 30.0
arcface_m = 0.5
# arcface_m = 1.0

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 畳み込み層1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 畳み込み層2
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全結合層1
        self.fc2 = nn.Linear(128, 128)  # 全結合層2（出力層）

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        # x = nn.functional.dropout(x, 0.25, training=self.training)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        # x = nn.functional.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x

class AdaCosLoss(nn.Module):
    def __init__(self, num_features, num_classes=10, m=arcface_m):
        super(AdaCosLoss, self).__init__()
        self.num_classes = num_classes  # クラス数
        self.s = math.sqrt(2) * math.log(num_classes - 1)  # 固定値のスケーリングファクター
        self.m = m  # マージンmを設定
        self.ce_loss = nn.CrossEntropyLoss()  # 交差エントロピー損失
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # デバイスを確認
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features).to(device))  # self.Wを適切なデバイスに配置
        nn.init.xavier_normal_(self.W)  # 重みの初期化

    def forward(self, logits, labels):
        # logitsをL2正規化
        logits = F.normalize(logits, p=2, dim=-1)  # L2正規化
        # 重みベクトルをL2正規化
        W = F.normalize(self.W, p=2, dim=-1)  # L2正規化
        # コサイン類似度の計算
        cos_theta = logits @ W.T  # 内積でコサイン類似度を計算
        # マージンmを適用
        phi = cos_theta + self.m  # マージンmを適用
        # print(f"AdaCos cos_theta: {cos_theta}, phi: {phi}")  # デバッグ情報を出力
        return self.ce_loss(self.s * phi, labels)  # スケーリングファクターとマージンを適用して損失を計算

class ArcFaceLoss(nn.Module):
    def __init__(self, num_features, num_classes=10, s=arcface_s, m=arcface_m):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # デバイスを確認
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features).to(device))  # self.Wを適切なデバイスに配置
        nn.init.xavier_normal_(self.W)

    def forward(self, logits, labels):
        logits = F.normalize(logits, p=2, dim=-1)
        W = F.normalize(self.W, p=2, dim=-1)
        cos_theta = logits @ W.T
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # デバイスを確認
        one_hot = torch.zeros(cos_theta.size(), device=device)  # one_hotを適切なデバイスに配置
        labels = labels.to(device)  # labelsも適切なデバイスに配置
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = cos_theta * 1.0
        output -= one_hot * self.m
        # print(f"ArcFace cos_theta: {cos_theta}, phi: {phi}")  # デバッグ情報を出力
        return F.cross_entropy(self.s * output, labels)

def train_and_evaluate(model, loss_fn, train_loader, test_loader, epochs=10):
    model = model.to('cuda')  # モデルをGPUに移動
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        # 訓練データでの性能評価
        correct_train = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to('cuda'), target.to('cuda')
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct_train += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {epoch + 1} Training accuracy: {100. * correct_train / len(train_loader.dataset)}%")
        # テストデータでの性能評価
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch + 1}: Test loss: {test_loss}, Test accuracy: {100. * correct / len(test_loader.dataset)}%")

# データの前処理を追加（正規化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNISTの平均と標準偏差
])

train_loader = DataLoader(datasets.MNIST('assets/MNIST', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST('assets/MNIST', train=False, download=True, transform=transform), batch_size=128)

# # CrossEntropyでの訓練と評価
# print("Training with CrossEntropy")
# model = SimpleCNN()  # モデルのインスタンスを作成
# criterion = nn.CrossEntropyLoss()  # CrossEntropy損失関数を設定
# train_and_evaluate(model, criterion, train_loader, test_loader, epochs=epoch)  # 訓練と評価を実行

# AdaCosでの訓練と評価
print("Training with AdaCos")
model = SimpleCNN()
train_and_evaluate(model, AdaCosLoss(num_features=128), train_loader, test_loader, epochs=epoch)  # num_featuresを128に変更

# ArcFaceでの訓練と評価
print("Training with ArcFace")
model = SimpleCNN()
train_and_evaluate(model, ArcFaceLoss(num_features=128, s=arcface_s, m=arcface_m), train_loader, test_loader, epochs=epoch)  # num_featuresを128に変更

