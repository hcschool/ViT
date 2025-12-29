# src/model.py
from transformers import ViTForImageClassification
from .config import Config

def get_model(label2id, id2label):
    model = ViTForImageClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )
    return model