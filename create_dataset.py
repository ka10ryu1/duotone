#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import cv2
import argparse
import numpy as np

import Tools.imgfunc as IMG
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('color',   help='使用する入力画像')
    parser.add_argument('duotone', help='使用する正解画像')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--round', '-r', type=int, default=100,
                        help='切り捨てる数（default: 100）')
    parser.add_argument('--augmentation', '-a', type=int, default=2,
                        help='水増しの種類（default: 2）')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def saveNPZ(x, y, name, folder, size):
    """
    入力データと正解データをNPZ形式で保存する
    [in] x:      保存する入力データ
    [in] y:      保存する正解データ
    [in] name:   保存する名前
    [in] folder: 保存するフォルダ
    [in] size:   データ（正方形画像）のサイズ
    """
    size_str = '_' + str(size).zfill(2) + 'x' + str(size).zfill(2)
    num_str = '_' + str(x.shape[0]).zfill(6)
    np.savez(F.getFilePath(folder, name + size_str + num_str), x=x, y=y)


def edgeDetect(duotone, color):
    """
    カラー画像のエッジを抽出し、モノクロ2値画像に重ねる
    [in] duotone: モノクロ2値画像
    [in] color:   エッジ抽出したいカラー画像
    [out] エッジとモノクロ2値を重ねた画像
    """

    # カーネルサイズの設定
    kernel5 = np.ones((5, 5), np.uint8)
    # 画像のノイズ除去
    gray = cv2.fastNlMeansDenoising(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), h=20)
    # dilate -> diff で線が抽出できる
    dilation = cv2.dilate(gray, kernel5, iterations=1)
    diff = cv2.subtract(dilation, gray)
    diff_inv = 255 - diff
    # 線を単色 -> 2値にする
    _, edge = cv2.threshold(diff_inv, 225, 255, cv2.THRESH_BINARY)
    # 線画とモノクロ2値を重ねる
    return cv2.bitwise_and(duotone, edge)


def main(args):

    # 入力のカラー画像を読み込む
    if IMG.isImgPath(args.color):
        print('color image read:\t', args.color)
        x = cv2.imread(args.color, IMG.getCh(3))
    else:
        print('[ERROR] color image not found:', args.color)
        exit()

    # 正解のモノクロ画像を読み込む
    if IMG.isImgPath(args.duotone):
        print('duotone image read:\t', args.duotone)
        y = edgeDetect(cv2.imread(args.duotone, IMG.getCh(1)), x)
    else:
        print('[ERROR] duotone image not found:', args.duotone)
        exit()

    # 入力画像と正解画像を
    # [1] 反転して増やす [flip()]
    # [2] 正方形に分割する [splitSQN()]
    print('split and rotate images...')
    x, _ = IMG.splitSQN(IMG.flip(x, args.augmentation),
                        args.img_size, args.round)
    y, _ = IMG.splitSQN(IMG.flip(y, args.augmentation),
                        args.img_size, args.round)

    # 画像の並び順をシャッフルするための配列を作成する
    # colorとmonoの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    print('shuffle images...')
    shuffle = np.random.permutation(range(len(x)))
    train_size = int(len(x) * args.train_per_all)
    print('train/all : {0}/{1}'.format(train_size, len(x)))
    # float16で保存することでファイルサイズを軽減できる
    # 情報量は落ちるが問題ないレベル
    dtype = np.float16
    train_x = IMG.imgs2arr(x[shuffle[:train_size]], dtype=dtype)
    train_y = IMG.imgs2arr(y[shuffle[:train_size]], dtype=dtype)
    test_x = IMG.imgs2arr(x[shuffle[train_size:]], dtype=dtype)
    test_y = IMG.imgs2arr(y[shuffle[train_size:]], dtype=dtype)
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    saveNPZ(train_x, train_y, 'train', args.out_path, args.img_size)
    saveNPZ(test_x, test_y, 'test', args.out_path, args.img_size)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
