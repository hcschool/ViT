import torch
import os 
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from src.config import Config

class ViTInference:
    def __init__(self, model_path=f"{Config.OUTPUT_DIR}/final_model"):
        
        print(f"모델을 로드 중입니다: {model_path}")
        
        if not os.path.exists(model_path): 
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}. 먼저 train.py를 실행하세요.") 

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.model.to(Config.DEVICE)
        self.model.eval() 

    def predict(self, image_path):
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(Config.DEVICE)

        with torch.no_grad(): 
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        labels = self.model.config.id2label
        prediction_label = labels[predicted_class_idx]
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()

        return prediction_label, confidence

if __name__ == "__main__":
    try:
        infer = ViTInference()
        test_img = "test_leaf.jpg" 
        
        if not os.path.exists(test_img): 
             print(f"주의: 테스트 이미지({test_img})가 없습니다.") 
        else:
            label, score = infer.predict(test_img)
            print(f"--- 추론 결과 ---")
            print(f"예측 결과: {label}")
            print(f"신뢰도: {score:.2%}")

    except Exception as e:
        print(f"오류 발생: {e}")