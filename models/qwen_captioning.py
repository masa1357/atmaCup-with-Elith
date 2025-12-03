from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

class QwenChat:
    def __init__(self, model: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ここで完全テキストモデルを指定する
        # 例: "Qwen/Qwen2.5-7B-Instruct" など
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True
        )

    def generate_chat(
        self,
        messages,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        messages: [{"role": "user"/"assistant"/"system", "content": "..."}, ...]
        の形式を想定
        """

        # Qwen 系の chat テンプレートを使ってプロンプトを構成
        # add_generation_prompt=True で最後にアシスタント発話の開始を付ける
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 生成部分だけを取り出す（プロンプトの長さ以降）
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # call_local_llm が期待している返り値形式に合わせる
        return {
            "message": {
                "role": "assistant",
                "content": generated_text
            }
        }

