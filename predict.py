#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルとモデルパラメータを利用して推論実行する'
#

import cv2
import time
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer.cuda import to_cpu

import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--out_rate', '-r', type=float, default=0.5,
                        help='生成後の画像の拡大率 [default: 0.5]')
    parser.add_argument('--batch', '-b', type=int, default=20,
                        help='ミニバッチサイズ [default: 20]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def predict(model, data, batch, org_shape, rate, gpu):
    """
    推論実行メイン部
    [in]  model:     推論実行に使用するモデル
    [in]  data:      分割（IMG.split）されたもの
    [in]  batch:     バッチサイズ
    [in]  org_shape: 分割前のshape
    [in]  rate:      出力画像の拡大率
    [in]  gpu:       GPU ID
    [out] img:       推論実行で得られた生成画像
    """

    # dataには圧縮画像と分割情報が含まれているので、分離する
    comp, size = data
    imgs = []
    st = time.time()
    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    for i in range(0, len(comp), batch):
        x = IMG.imgs2arr(comp[i:i + batch], gpu=gpu)
        y = model.predictor(x)
        imgs.extend(IMG.arr2imgs(to_cpu(y.array)))

    print('exec time: {0:.2f}[s]'.format(time.time() - st))
    # 生成された画像を分割情報をもとに結合する
    buf = [np.vstack(imgs[i * size[0]: (i + 1) * size[0]])
           for i in range(size[1])]
    img = np.hstack(buf)
    # 出力画像は入力画像の2倍の大きさになっているので半分に縮小する
    img = IMG.resize(img, 1 / rate)
    # 結合しただけでは画像サイズがオリジナルと異なるので切り取る
    return img[:org_shape[0], :org_shape[1]]


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    net, unit, ch, size, layer, sr, af1, af2 = GET.modelParam(args.param)
    # 学習モデルを生成する
    if net == 0:
        from Lib.network import JC_DDUU as JC
    else:
        from Lib.network2 import JC_UDUD as JC

    model = L.Classifier(
        JC(n_unit=unit, n_out=1, rate=sr, actfun1=af1, actfun2=af2)
    )

    # load_npzのpath情報を取得し、学習済みモデルを読み込む
    load_path = F.checkModelType(args.model)
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(F.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    else:
        model.to_intel64()

    # 高圧縮画像の生成
    org_imgs = [cv2.imread(name, IMG.getCh(ch))
                for name in args.jpeg if IMG.isImgPath(name)]
    imgs = []
    # 学習モデルを入力画像ごとに実行する
    for i, ei in enumerate(org_imgs):
        with chainer.using_config('train', False):
            img = predict(
                model, IMG.splitSQ(ei, size),
                args.batch, ei.shape, sr, args.gpu
            )

        imgs.append(IMG.resize(img, args.out_rate))

    for img in imgs:
        cv2.imshow('view', img)
        cv2.waitKey()
        # 生成結果を保存する
        name = F.getFilePath(
            args.out_path, 'comp-' + str(i * 10 + 1).zfill(3), '.jpg'
        )
        print('save:', name)
        cv2.imwrite(name, img)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
