# app.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from call_api import run_qwen_pipeline, get_last_log_line
import yaml

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool 
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/log_tail")
def log_tail():
    return get_last_log_line() or "（ログなし）"

@app.post("/run_pipeline")
async def run_pipeline(payload: dict):
    prompts = payload["prompts"]  # ["text1", "text2", ..., "text6"]

    # result_yaml = run_qwen_pipeline(prompts)
    result_yaml = await run_in_threadpool(run_qwen_pipeline, prompts)
    yaml_text = yaml.dump(result_yaml, allow_unicode=True)

    summary = result_yaml.get("スコアサマリー", {})
    attack = summary.get("攻撃性能", {})
    defense = summary.get("防御性能", {})
    unified = summary.get("統合スコア", {})

    score_data = {
        "attack": float(unified.get("攻撃パフォーマンス", 0)),
        "defense": float(unified.get("防御パフォーマンス", 0)),
        "final": float(unified.get("最終スコア", 0))
    }

    latest_log = get_last_log_line()

    return {
        "output": yaml_text,
        "scores": score_data,
        "latest_log": latest_log,
    }
