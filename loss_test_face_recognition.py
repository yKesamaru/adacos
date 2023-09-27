import math

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses, regularizers, samplers
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# ハイパーパラメータの設定
epoch = 10
num_classes = 16
arcface_s = 64.0
arcface_m = 28.6
batch_size = 16
lr = 0.01
resolution = 224
embedding_size = 512

# EfficientNetを読み込む
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        # self.model = timm.create_model("efficientnet_b0", pretrained=True)  # EfficientNet
        self.model = timm.create_model("efficientnetv2_rw_s", pretrained=True)  # EfficientNetV2
        num_features = self.model.classifier.in_features  # `_fc`属性を使用する
        # modelの最終層にembedding_size次元のembedder層を追加
        self.model.classifier = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        x = self.model(x)
        return x

def train(model, loss_fn, train_loader, test_loader, epochs=epoch):
    model = model.to('cuda')  # モデルをGPUに移動
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        correct_train = 0  # 追加; 正解数を初期化
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

class CustomSubset(Subset):
    def __init__(self, subset, transform=None):
        super(CustomSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# 画像の平均値と標準偏差
mean_value = [0.485, 0.456, 0.406]
std_value = [0.229, 0.224, 0.225]

# 画像の前処理設定（訓練用）
train_transform=transforms.Compose([
        transforms.Resize((resolution, resolution)),  # 解像度にリサイズ
        transforms.RandomHorizontalFlip(p=0.5),  # 水平方向にランダムに反転
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize(mean=mean_value, std=std_value)  # 正規化
    ])

# 画像の前処理設定（テスト用）
test_transform=transforms.Compose([
        transforms.Resize((resolution, resolution)),  # 解像度にリサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize(mean=mean_value, std=std_value)  # 正規化
    ])

# DataLoaderの設定
# 元のデータセットを読み込む
dataset = ImageFolder(root='assets/face_data')
# データセットのサイズを取得
dataset_size = len(dataset)
# 訓練データとテストデータの割合を設定（8:2）
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
# データセットを訓練用とテスト用に分割
train_subset, test_subset = random_split(dataset, [train_size, test_size])
# transformを適用
train_dataset = CustomSubset(train_subset, transform=train_transform)
test_dataset = CustomSubset(test_subset, transform=test_transform)
# DataLoaderの設定
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 訓練用
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # テスト用

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

# # DEBUG
# # ラベルとクラス名の対応関係を表示
# print("Label to class mapping:", dataset.class_to_idx)
# # DataLoaderからいくつかのサンプルを取得
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# # 画像とラベルを表示
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     img = images[i].permute(1, 2, 0).numpy()  # C,H,W -> H,W,Cに変換
#     img = img * std_value + mean_value  # 正規化を元に戻す
#     img = img.clip(0, 1)  # 0~1の範囲にクリッピング
#     plt.imshow(img)
#     plt.title(f'Label: {labels[i]}')
# plt.show()

def evaluate_with_cross_entropy(model, train_loader, test_loader, loss_fn):
    model.eval()  # 評価モード
    model = model.to('cuda')
    # 訓練データでの性能評価
    correct_train = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
    print(f"Training accuracy: {100. * correct_train / len(train_loader.dataset)}%")
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
    for data, target in train_loader:
        data, target = data.to('cuda'), target.to('cuda')
        with torch.no_grad():
            output = model(data)
        train_features.append(output.cpu().numpy())
        train_labels.append(target.cpu().numpy())
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)
    # k-NNモデルの訓練
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_labels)
    # 訓練データでの評価
    correct_train = 0
    total_train = 0
    for data, target in train_loader:
        data, target = data.to('cuda'), target.to('cuda')
        with torch.no_grad():
            output = model(data)
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        pred_train = knn.predict(output)
        correct_train += np.sum(pred_train == target)
        total_train += target.shape[0]
    train_accuracy = 100. * correct_train / total_train
    print(f"{loss_name}: Training accuracy: {train_accuracy}%")
    # テストデータでの評価
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
    test_accuracy = 100. * correct / total
    print(f"{loss_name}: Test accuracy: {test_accuracy}%")

# ArcFaceでの訓練と評価
print("Training with ArcFace")
model = EfficientNet()
R = regularizers.RegularFaceRegularizer()
arcface_loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=arcface_m, scale=arcface_s, weight_regularizer=R)
train(model, arcface_loss, train_loader, test_loader, epochs=epoch)
# ArcFaceでの評価
evaluate_with_margin('arcface', model, train_loader, test_loader)

# AdaCosでの訓練と評価
print("Training with AdaCos")
model = EfficientNet()
adacos_loss = AdaCosLoss(num_features=embedding_size, num_classes=num_classes, m=arcface_m)
train(model, adacos_loss, train_loader, test_loader, epochs=epoch)
# ArcFaceでの評価
evaluate_with_margin('adacos', model, train_loader, test_loader)

# CrossEntropyでの訓練
print("Training with CrossEntropy")
model = EfficientNet()
cross_entropy_loss = nn.CrossEntropyLoss()
train(model, cross_entropy_loss, train_loader, test_loader, epochs=epoch)
# CrossEntropyでの評価
evaluate_with_cross_entropy(model, train_loader, test_loader, cross_entropy_loss)

"""
Training with ArcFace
arcface: Training accuracy: 98.60643921191735%
arcface: Test accuracy: 85.22072936660268%
Training with AdaCos
adacos: Training accuracy: 99.42335415665545%
adacos: Test accuracy: 90.01919385796545%
Training with CrossEntropy
Training accuracy: 94.56991830850552%
Test loss: 0.022136613415579192, Test accuracy: 89.44337811900192%

| Loss Function  | Training Accuracy (%) | Test Accuracy (%) | Test Loss               |
|----------------|-----------------------|-------------------|-------------------------|
| ArcFace        | 98.61                 | 85.22             | N/A                     |
| AdaCos         | 99.42                 | 90.02             | N/A                     |
| CrossEntropy   | 94.57                 | 89.44             | 0.022136613415579192    |

"""