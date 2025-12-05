# FROM nvidia/cuda:12.5.0-base-ubuntu22.04
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python-is-python3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ライブラリをインストール
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install transformers_stream_generator

# アプリケーションファイルを追加
COPY app /app

EXPOSE 8000