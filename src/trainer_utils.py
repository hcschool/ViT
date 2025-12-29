import numpy as np
import evaluate

# 전역 로드
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision") 
recall_metric = evaluate.load("recall") 

def compute_metrics(eval_pred):
    """
    Trainer가 검증 단계에서 호출하는 함수
    Args:
        eval_pred: (predictions, labels) 튜플
    """
    logits, labels = eval_pred
    
    if isinstance(logits, tuple): 
        logits = logits[0] 
        
    predictions = np.argmax(logits, axis=-1)
    
    # 정확도 계산
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # F1, Precision, Recall 계산 (Macro average)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro") #
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro") #

    # 결과를 딕셔너리 형태로 반환
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"], 
        "recall": recall["recall"] 
    }