import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def detect_red_clothes(image_path):
    # 1. 載入 OpenAI 的預訓練 CLIP 模型與處理器
    # 第一次執行會自動下載模型 (約 600MB)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 2. 讀取圖片
    image = Image.open(image_path)

    # 3. 定義標籤 (對比式描述)
    # 第一個是目標，第二個是干擾項，這能幫助模型做二元排除
    labels = ["a person wearing red clothes", "a red object", "a photo of person", "an empty street or background"]

    # 4. 影像預處理與推理
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. 計算機率 (Softmax)
    # logits_per_image 是圖片與各個文字標籤的相似度分數
    probs = outputs.logits_per_image.softmax(dim=1)
    
    # 取得結果
    red_prob = probs[0][0].item()
    others_prob = probs[0][1].item()
    
    is_detected = red_prob > 0.55  # 設定一個信心門檻，0.7 是常用的精準值

    print(f"--- 檢測結果 ---")
    print(f"圖片路徑: {image_path}")
    print(f"偵測到紅衣人: {'Yes' if is_detected else 'No'}")
    print(f"信心指數: {red_prob:.2%}")

# 使用範例
if __name__ == "__main__":
    detect_red_clothes("data/test_images/images.jpg")
