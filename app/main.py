# app/main.py
import uvicorn

if __name__ == "__main__":
    # "app:app" = app.py 内の変数 app (FastAPI インスタンス)
    uvicorn.run("app:app", host="0.0.0.0", port=8000)