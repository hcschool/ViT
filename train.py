# train.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import torch
import pandas as pd
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, DefaultDataCollator, set_seed
from sklearn.metrics import confusion_matrix
from src.config import Config
from src.dataset import prepare_dataset
from src.model import get_model
from src.trainer_utils import compute_metrics

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

def print_and_save_epoch_table(history):
    """Epoch별 학습 기록을 표로 저장"""
    epoch_data = {}
    for log in history:
        if 'epoch' not in log: continue
        epoch = round(log['epoch'])
        if epoch == 0: continue
        if epoch not in epoch_data: epoch_data[epoch] = {'epoch': epoch}
        if 'loss' in log: epoch_data[epoch]['train_loss'] = log['loss']
        if 'eval_loss' in log:
            epoch_data[epoch]['val_loss'] = log['eval_loss']
            epoch_data[epoch]['accuracy'] = log.get('eval_accuracy', 0)
            epoch_data[epoch]['f1'] = log.get('eval_f1', 0)
            epoch_data[epoch]['precision'] = log.get('eval_precision', 0)
            epoch_data[epoch]['recall'] = log.get('eval_recall', 0)

    rows = []
    for e in sorted(epoch_data.keys()):
        data = epoch_data[e]
        if 'val_loss' in data:
            rows.append({
                "Epoch": f"{data['epoch']}",
                "Train Loss": f"{data.get('train_loss', 0):.4f}",
                "Val Loss": f"{data['val_loss']:.4f}",
                "Acc": f"{data['accuracy']:.4f}",
                "F1": f"{data['f1']:.4f}",
                "Prec": f"{data['precision']:.4f}",
                "Rec": f"{data['recall']:.4f}"
            })

    if not rows: return
    df = pd.DataFrame(rows)
    
    print("\nEpoch별 학습 요약표")
    print(df.to_string(index=False)) 

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("Training History", fontsize=14, fontweight="bold", pad=20)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    plt.savefig(f"{Config.OUTPUT_DIR}/history_table.png", bbox_inches='tight', dpi=300)
    print(f"학습 기록 표 저장 완료: {Config.OUTPUT_DIR}/history_table.png")
    plt.close()

def save_final_metrics(metrics):
    """최종 테스트 메트릭 저장"""
    data = []
    headers = ["Metric", "Value"]
    keys = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    display_names = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"]
    
    for key, name in zip(keys, display_names):
        if key in metrics:
            data.append([name, f"{metrics[key]:.4f}"])
            
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("Final Test Metrics", fontsize=14, fontweight="bold")
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    plt.savefig(f"{Config.OUTPUT_DIR}/final_metrics.png", bbox_inches='tight', dpi=300)
    print(f"최종 metric 표 저장 완료: {Config.OUTPUT_DIR}/final_metrics.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Confusion Matrix 저장"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/confusion_matrix.png")
    print(f"Confusion Matrix 저장 완료: {Config.OUTPUT_DIR}/confusion_matrix.png")
    plt.close()

def main():
    
    seed_everything(Config.SEED)
    print(f"SEED: {Config.SEED}")

    print(f"학습 장치(DEVICE): {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    dataset, id2label, label2id, processor = prepare_dataset()
    model = get_model(label2id, id2label)
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        eval_strategy=Config.EVAL_STRATEGY,
        save_strategy=Config.SAVE_STRATEGY,
        logging_strategy=Config.LOGGING_STRATEGY,
        load_best_model_at_end=Config.LOAD_BEST_MODEL_AT_END,      
        metric_for_best_model=Config.METRIC_FOR_BEST_MODEL,       
        greater_is_better=Config.GREATER_IS_BETTER,           
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        remove_unused_columns=Config.REMOVE_UNUSED_COLUMNS,
        dataloader_num_workers=Config.NUM_WORKERS,
        warmup_ratio=Config.WARMUP_RATIO,
        report_to=Config.REPORT_TO,
        seed=Config.SEED
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("\n학습을 시작합니다")
    trainer.train()
    
    print_and_save_epoch_table(trainer.state.log_history)
    
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"\n최고 성능 checkpoint: {best_ckpt}")
    
    # 최종 모델 저장 (./outputs/vit-base-beans/final_model)
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_model")
    
    print("\nTest set 평가 중")
    predictions = trainer.predict(dataset['test'])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    metrics = predictions.metrics
    
    save_final_metrics(metrics)
    
    class_names = [id2label[i] for i in range(len(id2label))]
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # 최종 리포트 터미널에 출력
    print("\n" + "="*40)
    print("           최종 결과 리포트           ")
    print("="*40)
    print(f"Test Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Test Macro F1 : {metrics['test_f1']:.4f}")
    print(f"Precision     : {metrics['test_precision']:.4f}")
    print(f"Recall        : {metrics['test_recall']:.4f}")
    print("="*40)
    print(f"모든 결과 파일은 {Config.OUTPUT_DIR} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
