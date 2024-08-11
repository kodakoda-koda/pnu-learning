# PNU Learning
[PNU Learningの検証と考察](https://kodakoda-koda.github.io/posts/PNULearning/)で使用したコード

## 実行方法
### 1. ライブラリをインストール
```console
~$ pip install -r requirements.txt
```

### 2. データをダウンロードし，data/に配置
[knowledgator/events_classification_biotech](https://huggingface.co/datasets/knowledgator/events_classification_biotech/tree/main)にアクセスし，train.csvとtest.csvをダウンロード．\
data/に配置する．

### 3. 実行
```console
~$ python ./src/main.py --unlabel_rate <アンラベル率> --eta <エータの値>
```