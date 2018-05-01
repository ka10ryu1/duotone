# 概要

任意のカラー画像をモノクロ2階調風画像に変換する。
過学習を起こすことが目的であるところがポイントの一つ。

## 学習結果

### 入力画像

<img src="https://github.com/ka10ryu1/ImageDataAll/blob/master/test-castle.JPG" width="640px">

<img src="https://github.com/ka10ryu1/ImageDataAll/blob/master/test-braggart.jpg" width="640px">

### 学習結果

<img src="https://github.com/ka10ryu1/duotone/blob/image/comp-001.jpg" width="640px">

<img src="https://github.com/ka10ryu1/duotone/blob/image/comp-002.jpg" width="640px">


**入力画像（上）と正解画像**

<img src="https://github.com/ka10ryu1/duotone/blob/image/npz2jpg_3.jpg" width="320px">

上記の結果は、過学習を起こしたモデルから生成されている。画風変換系はそれっぽく変換できていればいいので、過学習でも問題ない。

## デモの実行

以下を実行することで上記結果を再現できる。`demo.model`はファイルサイズが大きいのでgit-lfsで管理している。git-lfsを未インストールの場合はREADME最下部の**git-lfsのインストール**を参照すること。

```console
$ ./predict.py Model/* [なんらかのカラー画像]
```

# 動作環境

- **Ubuntu** 16.04.4 LTS ($ cat /etc/issue)
- **Python** 3.5.2 ($ python3 -V)
- **chainer** 4.0.0 ($ pip3 show chainer | grep Ver)
- **numpy** 1.14.2 ($ pip3 show numpy | grep Ver)
- **cupy** 4.0.0 ($ pip3 show cupy | grep Ver)
- **opencv-python** 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
.
├── LICENSE
├── Lib
│   ├── network.py  > jpegcompのネットワーク部分
│   ├── network2.py > duotoneのネットワーク部分
│   └── plot_report_log.py
├── Model
│   ├── demo.model
│   └── param.json
├── README.md
├── Tools
│   ├── LICENSE
│   ├── README.md
│   ├── Tests
│   │   ├── Lenna.bmp       > テスト用画像
│   │   ├── Mandrill.bmp    > テスト用画像
│   │   ├── test_getfunc.py > getfuncのテスト用コード
│   │   └── test_imgfunc.py > imgfuncのテスト用コード
│   ├── concat.py          > 複数の画像を任意の行列で結合する
│   ├── dot2png.py         > dot言語で記述されたファイルをPNG形式に変換する
│   ├── func.py            > 便利機能
│   ├── getfunc.py         > 各種パラメータ取得に関する便利機能
│   ├── imgfunc.py         > 画像処理に関する便利機能
│   ├── npz2jpg.py         > 作成したデータセット（.npz）の中身を画像として出力する
│   ├── plot_diff.py       > logファイルの複数比較
│   └── png_monitoring.py  > 任意のフォルダの監視
├── auto_train.sh
├── clean_all.sh
├── create_dataset.py > 画像を読み込んでデータセットを作成する
├── predict.py        > モデルとモデルパラメータを利用して推論実行する
├── predict_auto.py   > 任意のフォルダに保存された画像を推論実行し、別のフォルダに自動で保存する
└── train.py          > 学習メイン部
```

データセットはテストデータ含め[別リポジトリ](https://github.com/ka10ryu1/ImageDataAll)にて管理している。

# チュートリアル

## ファイル構成

以下のようなファイル構成を前提とする。また、作業はすべて`duotone`フォルダ直下で行う。

```console
.
├── ImageDataAll
└── duotone
```


## データセットを作成する

```console
$ ./create_dataset.py ../ImageDataAll/concat_color.jpg ./ImageDataAll/concat_duotone_2.png
```

`result`フォルダが作成され、その直下に以下の学習用データとテスト用データが保存作成される。データ量が大きい場合は、オプション引数`-a 0`を利用してデータ数を減らすといい。

- `test_32x32_*.npz`
- `train_32x32_*.npz`

## 学習する

以下で学習が開始される。

```console
$ ./train.py
```

基本的にGPU使用を推奨（`-g GPU_ID`）。その他オプション引数は`-h`で確認する。学習が終了すると`OUT_PATH`フォルダに以下が保存される。Topと同様のパラメータにしたい場合は`Model/param.json`を参考にすると良い。

- `*.json`      < 推論実行で使用するパラメータファイル
- `*.log`       < lossなどの学習時のデータ
- `*.model`     < 学習モデル
- `*.snapshot`  < 途中成果物
- `*_graph.dot` < ネットワーク
- `loss.png`    < lossのグラフ
- `lr.png`      < lrのグラフ

## 推論実行する

こちらもGPU必須だが、iDeep環境であれば1minかからず実行できる。

```console
$ ./predict.py result/*.model result/*.json ../ImageDataAll/test-castle.JPG -g 0
```

推論実行された結果が表示され、`OUT_PATH`フォルダ直下にもそれが保存される。

## 推論自動実行（おまけ）

任意のフォルダAに保存された画像を自動で取得し、推論実行を行い、任意のフォルダBに保存する（デモ用）。例えばこのフォルダABをDropboxなどのクラウドサービス上のフォルダに指定すればスマートフォンで撮影した画像を外出先で変換できる。

```console
./predict_auto.py result/*.model result/*.json ~/Dropbox/temp/duotone/monitor ~/Dropbox/temp/duotone/copy --force -g 0 -r 0.2
```

# git-lfsのインストール

まず、apt-getできるように設定する

```console
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```

※prixyありの場合は以下
```console
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo -E bash
```

準備ができたらapt-get実行

```console
$ sudo apt-get install git-lfs
```

git-lfsを初期化

```console
$ git lfs install
Updated git hooks.
Git LFS initialized.
```

ファイルをpullする

```console
$ git lfs pull
```