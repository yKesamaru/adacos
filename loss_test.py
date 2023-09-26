import math

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses, regularizers, samplers
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# ハイパーパラメータの設定
epoch = 30
num_classes = 10
arcface_s = 64.0
arcface_m = 30.0
batch_size = 16
lr = 0.01
resolution = 224
embedding_size = 512

# EfficientNetを読み込む
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        # self.model = timm.create_model("efficientnet_b0", pretrained=True)
        self.model = timm.create_model("efficientnetv2_rw_s", pretrained=True)
        num_features = self.model.classifier.in_features  # `_fc`属性を使用する
        # modelの最終層にembedding_size次元のembedder層を追加
        self.model.classifier = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        x = self.model(x)
        return x

def train_and_evaluate(model, loss_fn, train_loader, test_loader, epochs=epoch):
    model = model.to('cuda')  # モデルをGPUに移動
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Epochs"):  # 追加; tqdmでエポックの進捗を表示
        model.train()
        correct_train = 0  # 追加; 正解数を初期化
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            # 訓練データでのACC計算
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
        train_acc = 100. * correct_train / len(train_loader.dataset)  # 訓練データのACC
        # テストデータでのACC計算
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to('cuda'), target.to('cuda')  # データをGPUに移動
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(target.view_as(pred)).sum().item()
        test_acc = 100. * correct_test / len(test_loader.dataset)  # テストデータのACC
        tqdm.write(f"Epoch {epoch + 1}: Train ACC: {train_acc}%, Test ACC: {test_acc}%")  # 追加; tqdm.writeでACCを表示
        loss = loss_fn(output, target)
        print(f"Batch {batch_idx}, Loss: {loss.item()}")  # デバッグ出力

class CustomSubset(Subset):
    def __init__(self, subset, transform=None):
        super(CustomSubset, self).__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


# データの前処理を追加（正規化とリサイズ、チャンネル数調整）
transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),  # リサイズ
    transforms.Grayscale(num_output_channels=3),  # グレースケールをRGBに変換
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3チャンネルに対応
])

train_loader = DataLoader(datasets.MNIST('assets/MNIST', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('assets/MNIST', train=False, download=True, transform=transform), batch_size=batch_size)

# # 画像の平均値と標準偏差
# mean_value = [0.485, 0.456, 0.406]
# std_value = [0.229, 0.224, 0.225]

# # 画像の前処理設定（訓練用）
# train_transform=transforms.Compose([
#         transforms.Resize((resolution, resolution)),  # 解像度にリサイズ
#         transforms.RandomHorizontalFlip(p=0.5),  # 水平方向にランダムに反転
#         transforms.ToTensor(),  # テンソルに変換
#         transforms.Normalize(mean=mean_value, std=std_value)  # 正規化
#     ])

# # 画像の前処理設定（テスト用）
# test_transform=transforms.Compose([
#         transforms.Resize((resolution, resolution)),  # 解像度にリサイズ
#         transforms.ToTensor(),  # テンソルに変換
#         transforms.Normalize(mean=mean_value, std=std_value)  # 正規化
#     ])

# # DataLoaderの設定
# # 元のデータセットを読み込む
# dataset = ImageFolder(root='assets/face_data')
# # データセットのサイズを取得
# dataset_size = len(dataset)
# # 訓練データとテストデータの割合を設定（8:2）
# train_size = int(0.8 * dataset_size)
# test_size = dataset_size - train_size
# # データセットを訓練用とテスト用に分割
# train_subset, test_subset = random_split(dataset, [train_size, test_size])
# # transformを適用
# train_dataset = CustomSubset(train_subset, transform=train_transform)
# test_dataset = CustomSubset(test_subset, transform=test_transform)
# # DataLoaderの設定
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 訓練用
# test_loader = DataLoader(test_dataset, batch_size=batch_size)  # テスト用

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

# # CrossEntropyでの訓練と評価
# print("Training with CrossEntropy")
# model = EfficientNetV2()  # モデルのインスタンスを作成
# criterion = nn.CrossEntropyLoss()  # CrossEntropy損失関数を設定
# train_and_evaluate(model, criterion, train_loader, test_loader, epochs=epoch)  # 訓練と評価を実行

# ArcFaceでの訓練と評価
print("Training with ArcFace")
model = EfficientNet()  # EfficientNetV2のインスタンスを作成
R = regularizers.RegularFaceRegularizer()
arcface_loss = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=arcface_m, scale=arcface_s, weight_regularizer=R)  # ArcFaceの損失関数
train_and_evaluate(model, arcface_loss, train_loader, test_loader, epochs=epoch)

# AdaCosでの訓練と評価
print("Training with AdaCos")
model = EfficientNet()  # EfficientNetV2のインスタンスを作成
adacos_loss = AdaCosLoss(num_features=embedding_size, num_classes=num_classes, m=arcface_m)  # AdaCosの損失関数
train_and_evaluate(model, adacos_loss, train_loader, test_loader, epochs=epoch)
