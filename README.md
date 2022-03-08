# 事前準備
### Anacondaのインストール
https://www.anaconda.com/products/individual  
https://www.python.jp/install/anaconda/macos/install.html

### 必要なファイルのダウンロード
```python
git clone https://github.com/sk0926/rl_kensyu.git
cd rl_kensyu
conda env create -f environment.yml
conda activate rl_kensyu
```

### 動作確認
```python
python load.py
```
アニメーションが表示されればOK
