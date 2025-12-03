# app.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from call_api import run_qwen_pipeline
import yaml

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    # 超シンプルなフォーム（textarea 6個）
    return """
    <html>
      <body>
        <h1>Qwen YAML Generator</h1>
        <form action="/run" method="post">
          <p>prompt1:<br><textarea name="prompt1" rows="3" cols="80"></textarea></p>
          <p>prompt2:<br><textarea name="prompt2" rows="3" cols="80"></textarea></p>
          <p>prompt3:<br><textarea name="prompt3" rows="3" cols="80"></textarea></p>
          <p>prompt4:<br><textarea name="prompt4" rows="3" cols="80"></textarea></p>
          <p>prompt5:<br><textarea name="prompt5" rows="3" cols="80"></textarea></p>
          <p>prompt6:<br><textarea name="prompt6" rows="3" cols="80"></textarea></p>
          <button type="submit">送信</button>
        </form>
      </body>
    </html>
    """


@app.post("/run", response_class=PlainTextResponse)
def run(
    prompt1: str = Form(...),
    prompt2: str = Form(...),
    prompt3: str = Form(...),
    prompt4: str = Form(...),
    prompt5: str = Form(...),
    prompt6: str = Form(...),
):
    prompt_texts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6]

    yaml_dict = run_qwen_pipeline(prompt_texts)

    # dict → YAML 文字列
    yaml_str = yaml.safe_dump(yaml_dict, allow_unicode=True, sort_keys=False)
    return yaml_str
