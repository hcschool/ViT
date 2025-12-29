import streamlit as st
import torch
import random
import os
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification, AutoModel, AutoConfig
from src.config import Config
from src.dataset import prepare_dataset
from vis_utils import get_attention_map, overlay_attention

st.set_page_config(page_title="ViT Analysis", layout="wide")

st.markdown("""
<style>
    .header { font-size: 18px; font-weight: bold; text-align: center; margin-bottom: 10px; }
    .success { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
    .caption { text-align: center; color: gray; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    device = Config.DEVICE
    
    # 1. Fine-tuned Model
    ft_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    if not os.path.exists(ft_path):
        ft_model, ft_proc = None, None
    else:
        try:
            print(f"✅ 내 모델 로드: {ft_path}")
            ft_proc = AutoImageProcessor.from_pretrained(ft_path, use_fast=True)
            ft_model = ViTForImageClassification.from_pretrained(
                ft_path, 
                attn_implementation="eager" 
            ).to(device).eval()
        except Exception as e:
            print(f"FT Load Error: {e}")
            ft_model, ft_proc = None, None

    # 2. Pre-trained ViT
    base_tag = "google/vit-base-patch16-224-in21k"
    try:
        base_proc = AutoImageProcessor.from_pretrained(base_tag, use_fast=True)
        base_model = ViTForImageClassification.from_pretrained(
            base_tag, 
            attn_implementation="eager"
        ).to(device).eval()
    except Exception as e:
        base_model, base_proc = None, None

    # 3. InternViT (Flash Attention 비활성화)
    intern_tag = "OpenGVLab/InternViT-300M-448px"
    try:
        print(f"⏳ InternViT 로드 중 (Flash Attention 비활성화)...")
        intern_proc = AutoImageProcessor.from_pretrained(intern_tag, trust_remote_code=True, use_fast=False)
        
        # 모델 로드 전에 환경 변수 설정 (Flash Attention 비활성화)
        os.environ['INTERN_VIT_FORCE_NAIVE_ATTN'] = '1'
        
        # 모델 로드
        intern_model = AutoModel.from_pretrained(
            intern_tag, 
            trust_remote_code=True,
            torch_dtype=torch.float32  # float32 명시
        ).to(device)
        
        # 모델을 완전히 float32로 변환
        intern_model = intern_model.float()
        intern_model.eval()
        
        # Flash Attention 강제 비활성화 (모든 레이어 순회)
        flash_disabled_count = 0
        for name, module in intern_model.named_modules():
            # use_flash_attn 플래그 비활성화
            if hasattr(module, 'use_flash_attn'):
                module.use_flash_attn = False
                flash_disabled_count += 1
            
            # inner_attn이 FlashAttention이면 None으로 설정
            if hasattr(module, 'inner_attn'):
                module.inner_attn = None
                
            # qkv_bias 활성화 (안정성)
            if hasattr(module, 'qkv_bias'):
                module.qkv_bias = True
        
        print(f"  ✓ {flash_disabled_count}개 Attention 레이어 수정")
        print(f"  ✓ 모델 dtype: {next(intern_model.parameters()).dtype}")
        print(f"✅ InternViT 로드 완료 (Naive Attention 모드)")
        
    except Exception as e:
        print(f"❌ InternViT 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        intern_model, intern_proc = None, None

    # 4. Dataset
    dataset, id2label, _, _ = prepare_dataset(apply_transforms=False)
    test_ds = dataset['test']

    return ft_model, ft_proc, base_model, base_proc, intern_model, intern_proc, test_ds, id2label

def analyze_sample(sample, id2label, ft_model, ft_proc, base_model, base_proc, intern_model, intern_proc):
    img = sample['image'].convert("RGB")
    true_label = id2label[sample['labels']]
    
    # A. Pre-trained ViT
    viz_base = None
    if base_model:
        print("\n[Pre-trained ViT 처리 중...]")
        attn = get_attention_map(base_model, base_proc, img, Config.DEVICE)
        viz_base = overlay_attention(img, attn)
    else:
        viz_base = img

    # B. Fine-tuned ViT
    print("\n[Fine-tuned ViT 처리 중...]")
    inputs = ft_proc(img, return_tensors="pt").to(Config.DEVICE)
    with torch.no_grad():
        logits = ft_model(**inputs).logits
        pred_idx = logits.argmax(-1).item()
        conf = torch.nn.functional.softmax(logits, dim=-1)[0][pred_idx].item()
    pred_label = id2label[pred_idx]
    
    attn_ft = get_attention_map(ft_model, ft_proc, img, Config.DEVICE)
    viz_ft = overlay_attention(img, attn_ft)
    
    # C. InternViT
    viz_intern = None
    if intern_model:
        print("\n[InternViT 처리 중...]")
        attn = get_attention_map(intern_model, intern_proc, img, Config.DEVICE)
        if attn is not None:
            viz_intern = overlay_attention(img, attn)
            print("✅ InternViT 시각화 완료")
        else:
            viz_intern = img
            print("⚠️ InternViT attention 실패, 원본 이미지 사용")
    else:
        viz_intern = img
    
    return img, true_label, pred_label, conf, viz_base, viz_ft, viz_intern

# UI 구성
st.title("🌱 Vision Transformer Attention Analysis")

with st.spinner("모델 로딩 중..."):
    ft_model, ft_proc, base_model, base_proc, intern_model, intern_proc, test_ds, id2label = load_resources()

if ft_model is None:
    st.error("학습된 모델이 없습니다. train.py를 먼저 실행하세요.")
    st.stop()

# 모델 상태 표시
col1, col2, col3 = st.columns(3)
with col1:
    status = "✅" if base_model else "❌"
    st.info(f"{status} Pre-trained ViT")
with col2:
    status = "✅" if ft_model else "❌"
    st.info(f"{status} Fine-tuned ViT")
with col3:
    status = "✅" if intern_model else "❌"
    st.info(f"{status} InternViT-300M")

if st.button("🎲 클래스별 1개씩 분석 시작", type="primary"):
    
    # 클래스별 인덱스 찾기
    labels = test_ds['labels']
    class_0_indices = [i for i, label in enumerate(labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(labels) if label == 1]
    class_2_indices = [i for i, label in enumerate(labels) if label == 2]

    # 랜덤 선택
    selected_indices = []
    if class_0_indices: selected_indices.append(random.choice(class_0_indices))
    if class_1_indices: selected_indices.append(random.choice(class_1_indices))
    if class_2_indices: selected_indices.append(random.choice(class_2_indices))
    
    # 헤더
    h1, h2, h3, h4, h5 = st.columns([1, 1, 1, 1, 1])
    h1.markdown("<div class='header'>Original</div>", unsafe_allow_html=True)
    h2.markdown("<div class='header'>① Pre-trained ViT</div>", unsafe_allow_html=True)
    h3.markdown("<div class='header'>② Fine-tuned ViT</div>", unsafe_allow_html=True)
    h4.markdown("<div class='header'>③ InternViT</div>", unsafe_allow_html=True)
    h5.markdown("<div class='header'>Diagnostic</div>", unsafe_allow_html=True)
    st.divider()

    for idx in selected_indices:
        sample = test_ds[idx]
        img, true_lbl, pred_lbl, conf, viz_base, viz_ft, viz_intern = analyze_sample(
            sample, id2label, ft_model, ft_proc, base_model, base_proc, intern_model, intern_proc
        )
        
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        
        c1.image(img, width="stretch")
        c1.markdown(f"<div class='caption'>{true_lbl}</div>", unsafe_allow_html=True)
        
        c2.image(viz_base, width="stretch")
        c3.image(viz_ft, width="stretch")
        c4.image(viz_intern, width="stretch")
        
        with c5:
            if true_lbl == pred_lbl:
                st.markdown(f"<p class='success'>✅ Correct</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='fail'>❌ Wrong</p>", unsafe_allow_html=True)
            st.write(f"**Pred:** {pred_lbl}")
            st.write(f"**Conf:** {conf:.1%}")
        
        st.divider()
    
    st.success("✅ 분석 완료!")