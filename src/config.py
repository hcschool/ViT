import os
import torch

class Config:
    
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))

    # 경로 및 데이터
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/vit-base-beans")
    DATASET_NAME = os.environ.get("DATASET_NAME", "beans")
    MODEL_NAME = os.environ.get("MODEL_NAME", "google/vit-base-patch16-224-in21k") 
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 224))

    # param
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
    EPOCHS = int(os.environ.get("EPOCHS", 10)) 
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5)) 
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.01)) 
    WARMUP_RATIO = 0.1

    # Trainer 전략 설정 (TrainingArguments)
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOGGING_STRATEGY = "epoch"
    
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "f1"
    GREATER_IS_BETTER = True
    SAVE_TOTAL_LIMIT = 1
    REMOVE_UNUSED_COLUMNS = False
    REPORT_TO = "none" 

    EARLY_STOPPING_PATIENCE = 3