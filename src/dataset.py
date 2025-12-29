# src/dataset.py
from datasets import load_dataset
from transformers import AutoImageProcessor
from src.config import Config
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

def prepare_dataset(apply_transforms=True): 
    """
    Args:
        apply_transforms (bool): True면 모델 학습용 전처리 적용, False면 원본 데이터 반환
    """
    # 데이터셋 로드
    dataset = load_dataset(Config.DATASET_NAME)

    # 라벨 매핑
    label_feature = dataset["train"].features["labels"]
    if hasattr(label_feature, "names") and label_feature.names is not None:
        names = label_feature.names
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
    else:
        unique_ids = sorted(set(dataset["train"]["labels"]))
        id2label = {i: str(i) for i in unique_ids}
        label2id = {str(i): i for i in unique_ids}

    # 이미지 프로세서 로드
    processor = AutoImageProcessor.from_pretrained(Config.MODEL_NAME)
    
    # apply_transforms가 True일 때만 정의 및 적용
    if apply_transforms: 
        image_mean = processor.image_mean
        image_std = processor.image_std
        size = processor.size["height"]

        transform = Compose([
            Resize((size, size)),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ])

        def transforms_fn(examples):
            examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
            del examples["image"]
            return examples

        # 데이터셋에 적용
        if "train" in dataset:
            dataset["train"] = dataset["train"].with_transform(transforms_fn)
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].with_transform(transforms_fn)
        if "test" in dataset:
            dataset["test"] = dataset["test"].with_transform(transforms_fn)

    return dataset, id2label, label2id, processor