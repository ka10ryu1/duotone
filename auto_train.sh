#!/bin/bash
# auto_train.sh
# train.pyをいろいろな条件で試したい時のスクリプト
# train.pyの引数を手入力するため、ミスが発生しやすい。
# auto_train.shを修正したら、一度-cオプションを実行してミスがないか確認するべき

# オプション引数を判定する部分（変更しない）

usage_exit() {
    echo "Usage: $0 [-c]" 1>&2
    echo " -c: 設定が正常に動作するか確認する"
    exit 1
}

FLAG_CHK=""
while getopts ch OPT
do
    case $OPT in
        c)  FLAG_CHK="--only_check"
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

# 以下自由に変更する部分（オプション引数を反映させるなら、$FLG_CHKは必要）

COUNT=1

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -o ./result/003/ -b 10 -e 80 -u 64 -g 0 -opt ada_d $FLAG_CHK
./Tools/plot_diff.py ./result/0* --no
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -o ./result/003/ -b 10 -e 80 -u 64 -g 0 -opt ada_g $FLAG_CHK
./Tools/plot_diff.py ./result/0* --no
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -o ./result/003/ -b 10 -e 80 -u 64 -g 0 -opt n_ag $FLAG_CHK
./Tools/plot_diff.py ./result/0* --no
COUNT=$(( COUNT + 1 ))
