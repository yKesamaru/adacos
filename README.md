![](https://raw.githubusercontent.com/yKesamaru/adacos/master/assets/eye_catch.png)

- [はじめに](#はじめに)
- [結論](#結論)
- [背景](#背景)
  - [ハイパーパラメーターが嫌い](#ハイパーパラメーターが嫌い)
  - [ハイパーパラメーターフリー、かつ高性能ときいて](#ハイパーパラメーターフリーかつ高性能ときいて)
  - [疑問](#疑問)
- [`Fixed AdaCos`](#fixed-adacos)
- [実装](#実装)
  - [`loss_test.py`](#loss_testpy)
    - [出力結果](#出力結果)
  - [`loss_test_face_recognition.py`](#loss_test_face_recognitionpy)
    - [出力結果](#出力結果-1)
- [まとめ](#まとめ)
- [参考文献](#参考文献)

## はじめに
皆さん、損失関数は何を使ってらっしゃいますか？
不良品検出、顔認証など`open set recognition problem`が絡むタスクでは、`ArcFace`がよく使われている印象を受けます。
でも、`ArcFace`にはスケールファクターとマージンのハイパーパラメーターがあり、煩雑です。
`AdaCos`の論文に載っている固定スケールファクターを使えば、少なくともスケーリングファクターを調整する必要はなくなります。
しかし、`Fixed AdaCos`の精度と処理時間は、どんな感じなんでしょう。
そこで実際に実装して、試してみました。

## 結論
`AdaCos`を固定スケールで使うと、同等の処理時間で`ArcFace`よりも高い精度が出ました。

| Loss Function  | Training Accuracy (%) | Test Accuracy (%) | Test Loss               |
|----------------|-----------------------|-------------------|-------------------------|
| ArcFace        | 98.61                 | 85.22             | N/A                     |
| **AdaCos**     | **99.42**             | **90.02**         | N/A                     |
| CrossEntropy   | 94.57                 | 89.44             | 0.022136613415579192    |

- dataset
  - 16 directories, 2602 files
    ![](https://raw.githubusercontent.com/yKesamaru/adacos/master/assets/2023-09-27-12-17-05.png)

## 背景
### ハイパーパラメーターが嫌い
経験と勘がすべてを支配する「ハイパーパラメーター指定」。これは、機械学習において、**避けて通りたい**作業です。
ハイパーパラメーターの調整は、勘を外せばモデルの訓練が不安定になり、精度にも悪影響がでます。
自動調整するライブラリもあるにはありますが、時間もかかりますし、パラメーター範囲の指定も必要です。

### ハイパーパラメーターフリー、かつ高性能ときいて
「`AdaCos`はハイパーパラメーターフリーで、訓練過程で自動的にスケールパラメーターを調整できるだけでなく、高い顔認識精度を達成することが可能」、とききました。

そんな美味しい話があるんですね。

そこで早速、`AdaCos`を実装してtrainを回してみました。
しかし、なかなか精度が上がりません。どうしてだ。

そんな時は、論文を読むしかありません。
下のグラフを見てください。
![](https://raw.githubusercontent.com/yKesamaru/adacos/master/assets/2023-09-23-22-46-46.png)
このグラフを見ると、`AdaCos`において、同じクラスのコサイン類似度は大きく、逆に異なるクラスのコサイン類似度がグラフ中もっとも小さいことがわかります。

注目すべきは横軸です。
**`ArcFace`より`AdaCos`が優れた値を示すのは、2万epoch以降です**。
いや、横軸がepochとは書いてません。ミニバッチかもしれない。それでもエグい数字です。

これだったら`ArcFace`の方がお手軽ではないかと思いました。

というわけで解散！…でもいいんですけど、`AdaCos`の論文には「固定スケーリングファクター」なるものも紹介されています。それを試してみてからでも遅くはありません。

### 疑問
- 固定スケール`AdaCos`（`Fixed AdaCos`）の、イテレーションにおける縦軸がどこにも書いてない。
![](https://raw.githubusercontent.com/yKesamaru/adacos/master/assets/2023-09-24-10-12-49.png)

- 固定スケールファクターならば、`ArcFace`と同じepoch数でいけるんじゃないか？

## `Fixed AdaCos`
[論文](https://arxiv.org/pdf/1905.00292.pdf)より。
$$
\tilde{s}_f = \sqrt{2} \cdot \log(C - 1)
$$

クラス数が $16$ だとすると、 $約3.829$ 。

## 実装
2パターンを用意しました。
- `loss_test.py`
  - dataset
    - MNIST
      - 手書き数字の10クラス分類問題。やさしい。
  - model
    - 独自実装
    - 10 epoch
  - 損失関数比較
    - `ArcFace`: `regularizers.CenterInvariantRegularizer()`で正規化
    - `Simple_ArcFace`: `ArcFace`のみ
    - `Fixed AdaCoss`: スケーリングファクターを固定
    - `Cross Entropy`: クロスエントロピーロス
- `loss_test_face_recognition.py`
  - dataset
    - 16 directories, 2602 face image files
      - 16クラスの顔認証問題。比較的難しい。
  - model
    - EfficientNetV2-b0
    - 10 epoch
  - 損失関数比較
    - `ArcFace`: `regularizers.RegularFaceRegularizer()`で正規化
    - `Simple_ArcFace`: `ArcFace`のみ
    - `Fixed AdaCoss`: スケーリングファクターを固定
    - `Cross Entropy`: クロスエントロピーロス

| Script Name                 | Dataset Description           | Model          | Epochs | Loss Function Description                    |
|-----------------------------|-------------------------------|----------------|--------|---------------------------------------------|
| `loss_test.py`              | MNIST (10-class, Easy)        | Custom         | 10     | ArcFace with `regularizers.CenterInvariantRegularizer()`, Simple_ArcFace, Fixed AdaCos, Cross Entropy |
| `loss_test_face_recognition.py` | 16 dirs, 2602 face images (16-class, Hard) | EfficientNetV2-b0 | 10  | ArcFace with `regularizers.RegularFaceRegularizer()`, Simple_ArcFace, Fixed AdaCos, Cross Entropy  |


### `loss_test.py`
https://github.com/yKesamaru/adacos/blob/03c3ce8eef05ad178456ff2f16301b81c0cba3a9/loss_test.py#L1-L211

#### 出力結果

| Loss Function  | Training Accuracy (%) | Test Accuracy (%) | Test Loss               |
|----------------|-----------------------|-------------------|-------------------------|
| ArcFace        | 99.90                 | 97.36             | N/A                     |
| Simple_ArcFace | 98.72                 | 95.78             | N/A                     |
| AdaCos         | **99.95**                 | 98.09             | N/A                     |
| CrossEntropy   | 98.32                 | **98.17**             | 0.0004521963403734844   |

簡単な分類問題を、単純なネットワークで解決した場合、`CrossEntropy Loss`でも良い結果を残しています。
しかし、`Fixed AdaCos`も良いスコアです。

### `loss_test_face_recognition.py`
https://github.com/yKesamaru/adacos/blob/03c3ce8eef05ad178456ff2f16301b81c0cba3a9/loss_test_face_recognition.py#L1-L236

#### 出力結果

| Loss Function  | Training Accuracy (%) | Test Accuracy (%) | Test Loss               |
|----------------|-----------------------|-------------------|-------------------------|
| ArcFace        | 98.61                 | 85.22             | N/A                     |
| AdaCos         | **99.42**                 | **90.02**             | N/A                     |
| CrossEntropy   | 94.57                 | 89.44             | 0.022136613415579192    |

`open set recognition problem`の場合、損失の正則化を行った`ArcFace`よりも、単純な実装の`Fixed AdaCos`のスコアが勝りました。

## まとめ
`AdaCos`の固定スケーリングファクターは使えます。
今回は10 epoch程度の小規模な実験でしたが、実務にも役立つ知見を得られました。

以上です。ありがとうございました。

## 参考文献
[AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/pdf/1905.00292.pdf)