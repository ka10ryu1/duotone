#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '任意のフォルダに保存された画像を推論実行し、別のフォルダに自動で保存する'
#

import os
import cv2
import time
import argparse

import chainer
import chainer.links as L

from watchdog.observers import Observer

import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F
from predict import predict


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('monitor', help='監視するフォルダ')
    parser.add_argument('copy', help='コピーするフォルダ')
    parser.add_argument('--img_rate', '-r', type=float, default=0.5,
                        help='画像サイズの倍率 [default: 0.5]')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    parser.add_argument('--force', action='store_true',
                        help='monotorとcopyのフォルダがない場合に強制的に作成する')
    return parser.parse_args()


class JPGMonitor(F.ChangeHandler):
    def __init__(self, model, size, rate, batch, gpu, copy):
        self.model = model
        self.size = size
        self.rate = rate
        self.batch = batch
        self.gpu = gpu
        self.copy = copy

    def on_modified(self, event):
        path, name, ext = super().on_modified(event)
        if('jpg' in ext.lower()):
            print('duotone:{0}'.format(os.path.join(self.copy, path)))
            time.sleep(1)
            # 学習モデルを入力画像ごとに実行する
            with chainer.using_config('train', False):
                img = IMG.resize(cv2.imread(path, IMG.getCh(3)), self.rate)
                img = predict(self.model, IMG.splitSQ(img, self.size),
                              self.batch, img.shape, self.gpu)
                cv2.imwrite(os.path.join(self.copy, name), img)


def run(model, size, rate, batch, gpu, monitor, copy):
    while 1:
        event_handler = JPGMonitor(model, size, rate, batch, gpu, copy)
        observer = Observer()
        observer.schedule(event_handler, monitor, recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()


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
    # load_npzのpath情報を取得する
    load_path = F.checkModelType(args.model)
    # 学習済みモデルの読み込み
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

    print('Monitoring :', args.monitor)
    print('Copy to :', args.copy)
    print('Exit: Ctrl-c')
    run(model, size, args.img_rate, args.batch,
        args.gpu, args.monitor, args.copy)


if __name__ == '__main__':
    args = command()

    if not os.path.isdir(args.monitor):
        if args.force:
            os.makedirs(args.monitor)
        else:
            print('[Error] monitor folder not found:', args.monitor)
            exit()

    if not os.path.isdir(args.copy):
        if args.force:
            os.makedirs(args.copy)
        else:
            print('[Error] copy folder not found:', args.copy)
            exit()

    main(args)
