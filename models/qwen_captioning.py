from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

class QwenVLImageCaptioning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # モデルとトークナイザをロード
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL", device_map="auto", trust_remote_code=True
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

    def generate_caption(self, image_path):
        # ローカル画像のファイルパスをそのまま使用
        query = self.tokenizer.from_list_format([
            {'image': image_path},  # ファイルパスを直接指定
            {'text': 'Generate the caption in English with grounding:'},
        ])
        
        # トークン化し、モデルのデバイスに送る
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        
        # モデルでキャプションを生成
        pred = self.model.generate(**inputs)
        
        # 出力結果をデコード
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        
        # キャプション内の画像ボックス情報を取得
        caption_image = self.tokenizer.draw_bbox_on_latest_picture(response)
        
        # 結果を保存または表示
        if caption_image:
            caption_image.save("output_with_bbox.jpg")
            print("Bounding box image saved as 'output_with_bbox.jpg'")
        else:
            print("No bounding box detected in the image.")
        
        return response