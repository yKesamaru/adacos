import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses, regularizers, samplers
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ハイパーパラメータの設定
epoch = 50
arcface_s = 64.0
arcface_m = 28.6
num_classes = 10
batch_size = 128
embedding_size = 128
lr = 0.01

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 畳み込み層1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 畳み込み層2
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全結合層1
        self.fc2 = nn.Linear(128, embedding_size)  # 全結合層2（出力層）

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.dropout(x, 0.25, training=self.training)
        x = x.view(-1, 64 * 12 * 12)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x

class AdaCosLoss(nn.Module):
    def __init__(self, num_features, num_classes=num_classes, m=arcface_m):
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
    def __init__(self, num_features, num_classes=num_classes, s=arcface_s, m=arcface_m):
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

def train(model, loss_fn, train_loader, test_loader, epochs=epoch):
    model = model.to('cuda')  # モデルをGPUに移動
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

def evaluate_with_cross_entropy(model, train_loader, test_loader, loss_fn):
    model.eval()  # 評価モード
    model = model.to('cuda')
    # 訓練データでの性能評価
    # correct_train = 0
    # with torch.no_grad():
    #     for data, target in train_loader:
    #         data, target = data.to('cuda'), target.to('cuda')
    #         output = model(data)
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct_train += pred.eq(target.view_as(pred)).sum().item()
    # print(f"Training accuracy: {100. * correct_train / len(train_loader.dataset)}%")
    # テストデータでの性能評価
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
    print(f"Test loss: {test_loss}, Test accuracy: {100. * correct / len(test_loader.dataset)}%")

def evaluate_with_margin(loss_name, model, train_loader, test_loader):
    model.eval()  # 評価モード
    model = model.to('cuda')
    # 訓練データで特徴量を抽出
    train_features = []
    train_labels = []
    correct_train = 0
    # total_train = 0
    # for data, target in train_loader:
    #     data, target = data.to('cuda'), target.to('cuda')
    #     with torch.no_grad():
    #         output = model(data)
    #     train_features.append(output.cpu().numpy())
    #     train_labels.append(target.cpu().numpy())
    #     # 訓練データでの正確性を計算
    #     pred_train = output.argmax(dim=1, keepdim=True)
    #     correct_train += pred_train.eq(target.view_as(pred_train)).sum().item()
    #     total_train += target.shape[0]
    # train_features = np.vstack(train_features)
    # train_labels = np.concatenate(train_labels)
    # # 訓練データでの正確性を出力
    # print(f"{loss_name}: Training accuracy: {100. * correct_train / total_train}%")
    # k-NNモデルの訓練
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_labels)
    # テストデータで評価
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to('cuda'), target.to('cuda')
        with torch.no_grad():
            output = model(data)
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        pred = knn.predict(output)
        correct += np.sum(pred == target)
        total += target.shape[0]
    accuracy = 100. * correct / total
    print(f"{loss_name}: test accuracy with {loss_name}: {accuracy}%")# データの前処理を追加（正規化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNISTの平均と標準偏差
])

train_loader = DataLoader(datasets.MNIST('assets/MNIST', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('assets/MNIST', train=False, download=True, transform=transform), batch_size=batch_size)

# ArcFaceでの訓練
print("Training with ArcFace")
model = SimpleCNN()
R = regularizers.CenterInvariantRegularizer()
arcface_loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=arcface_m, scale=arcface_s, weight_regularizer=R)  # ArcFaceの損失関数
train(model, arcface_loss, train_loader, test_loader, epochs=epoch)

# ArcFaceでの評価
evaluate_with_margin('arcface', model, train_loader, test_loader)

# # ArcFaceでの訓練
# print("Training with ArcFace")
# model = SimpleCNN()
# train(model, ArcFaceLoss(num_features=embedding_size, s=arcface_s, m=arcface_m), train_loader, test_loader, epochs=epoch)

# AdaCosでの訓練
print("Training with AdaCos")
model = SimpleCNN()
train(model, AdaCosLoss(num_features=embedding_size), train_loader, test_loader, epochs=epoch)

# ArcFaceでの評価
evaluate_with_margin('adacos', model, train_loader, test_loader)

# CrossEntropyでの訓練
print("Training with CrossEntropy")
model = SimpleCNN()  # モデルのインスタンスを作成
criterion = nn.CrossEntropyLoss()  # CrossEntropy損失関数を設定
train(model, criterion, train_loader, test_loader, epochs=epoch)  # 訓練を実行

# CrossEntropyでの評価
evaluate_with_cross_entropy(model, train_loader, test_loader, criterion)  # 評価を実行
