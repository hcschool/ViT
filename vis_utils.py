import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import traceback

def get_attention_map(model, processor, image, device):
    """
    모델별로 최적화된 attention 추출
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # InternViT 감지
    model_name = model.__class__.__name__.lower()
    is_internvit = 'intern' in model_name
    
    print(f"  모델 타입: {model.__class__.__name__}")
    
    # 1. 표준 ViT 방식 시도
    attentions = None
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            attentions = outputs.attentions
            print(f"✅ 표준 방식으로 attention 추출 성공")
    except Exception as e:
        print(f"⚠️ 표준 방식 실패: {type(e).__name__} - {str(e)}")

    # 2. InternViT 전용 처리 (Naive Attention 모드)
    if attentions is None and is_internvit:
        print("🔍 InternViT 전용 방식 시도...")
        return _get_internvit_attention(model, inputs, device)

    # 3. Hook 방식
    if attentions is None:
        print("🔍 Hook 방식 시도 중...")
        attentions = _get_attention_via_hook(model, inputs)

    # 4. 처리 실패
    if attentions is None:
        print("❌ 모든 추출 방법 실패")
        return None

    # Attention 후처리
    try:
        if isinstance(attentions, (tuple, list)):
            last_attn = attentions[-1]
        else:
            last_attn = attentions

        # Shape 확인
        print(f"  Attention shape: {last_attn.shape}")
        
        # Shape 처리
        if last_attn.ndim == 4:  # (B, H, N, N)
            attn_map = last_attn[0, :, 0, 1:].mean(dim=0)
        elif last_attn.ndim == 3:  # (B, N, N)
            attn_map = last_attn[0, 0, 1:]
        else:
            print(f"⚠️ 예상치 못한 shape: {last_attn.shape}")
            return None
        
        # 그리드 변환
        num_patches = attn_map.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            grid_size = int(np.sqrt(num_patches))
            attn_map = attn_map[:grid_size*grid_size]
        
        attn_map = attn_map.reshape(grid_size, grid_size).cpu().numpy()
        
        # 정규화
        min_val, max_val = attn_map.min(), attn_map.max()
        if max_val - min_val > 1e-8:
            attn_map = (attn_map - min_val) / (max_val - min_val)
        else:
            attn_map = np.zeros_like(attn_map)
        
        attn_map = attn_map ** 2
        return attn_map
        
    except Exception as e:
        print(f"⚠️ 후처리 에러: {e}")
        traceback.print_exc()
        return None

def _get_internvit_attention(model, inputs, device):
    """
    InternViT 전용: Naive Attention에서 직접 추출
    """
    try:
        pixel_values = inputs['pixel_values']
        print(f"  입력 shape: {pixel_values.shape}")
        
        # Hook으로 attention weight 캡처
        captured_attn = []
        
        def attn_hook(module, input, output):
            # InternVisionAttention의 _naive_attn 메서드 내부에서
            # attention weight를 캡처
            if hasattr(module, '__class__') and 'Attention' in module.__class__.__name__:
                # output이 attention이면 저장
                if isinstance(output, torch.Tensor):
                    captured_attn.append(output.detach())
        
        # Attention 레이어에 hook 등록
        handles = []
        for name, module in model.named_modules():
            if 'attn' in name.lower() and 'InternAttention' in module.__class__.__name__:
                # attn 모듈의 내부 메서드에 접근
                handle = module.register_forward_hook(attn_hook)
                handles.append(handle)
                print(f"  ✓ Hook 등록: {name}")
        
        # Forward with hooks
        outputs = None
        try:
            with torch.no_grad():
                outputs = model(pixel_values)
        except Exception as e:
            print(f"  Forward 중 에러: {type(e).__name__} - {str(e)[:100]}")
            traceback.print_exc()
        finally:
            for h in handles:
                h.remove()
        
        # Captured attention 처리
        if captured_attn:
            print(f"  ✅ {len(captured_attn)}개 attention 캡처")
            # 마지막 레이어 사용
            last_attn = captured_attn[-1]
            
            # Shape에 따라 처리
            if last_attn.ndim == 4:  # (B, H, N, N)
                attn_map = last_attn[0, :, 0, 1:].mean(dim=0)
            elif last_attn.ndim == 3:  # (B, N, N)
                attn_map = last_attn[0, 0, 1:]
            elif last_attn.ndim == 2:  # (N, N)
                attn_map = last_attn[0, 1:]
            else:
                print(f"  ⚠️ 알 수 없는 attention shape: {last_attn.shape}")
                if outputs is not None:
                    return _get_internvit_similarity(outputs)
                return None
            
            num_patches = attn_map.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size != num_patches:
                grid_size = int(np.sqrt(num_patches))
                attn_map = attn_map[:grid_size*grid_size]
            
            attn_map = attn_map.reshape(grid_size, grid_size).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = attn_map ** 2
            
            return attn_map
        
        # Hook 실패 시 유사도 방식
        print("  Hook 실패, 유사도 방식으로 전환...")
        if outputs is not None:
            return _get_internvit_similarity(outputs)
        else:
            print("  ❌ outputs도 None, 최종 실패")
            return None
            
    except Exception as e:
        print(f"❌ InternViT 전용 처리 실패: {e}")
        traceback.print_exc()
        return None

def _get_internvit_similarity(outputs):
    """
    InternViT: Hidden states에서 CLS 토큰 유사도로 attention 근사
    """
    try:
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            print(f"  Hidden state shape: {hidden.shape}")
            
            # CLS와 패치 간 유사도
            cls_token = hidden[:, 0:1, :]
            patch_tokens = hidden[:, 1:, :]
            
            cls_norm = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-8)
            patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-8)
            
            similarity = torch.matmul(cls_norm, patch_norm.transpose(-2, -1))
            attn_map = similarity.squeeze(0).squeeze(0)
            
            num_patches = attn_map.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size != num_patches:
                attn_map = attn_map[:grid_size*grid_size]
            
            attn_map = attn_map.reshape(grid_size, grid_size).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_map = attn_map ** 2
            
            print("✅ CLS 유사도로 attention 근사 성공")
            return attn_map
        else:
            print(f"  ❌ last_hidden_state 없음")
            return None
    except Exception as e:
        print(f"❌ 유사도 계산 실패: {e}")
        return None

def _get_attention_via_hook(model, inputs):
    """Hook 기반 추출 (일반 ViT용)"""
    captured_data = []

    def attention_hook(module, input, output):
        try:
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                captured_data.append(output.detach())
            
            if isinstance(input, tuple):
                for inp in input:
                    if isinstance(inp, torch.Tensor) and inp.ndim == 4:
                        captured_data.append(inp.detach())
        except:
            pass

    handles = []
    
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ['attn', 'attention']):
            if isinstance(module, nn.Dropout):
                handles.append(module.register_forward_hook(attention_hook))

    print(f"  → {len(handles)}개 레이어에 Hook 설치")

    try:
        with torch.no_grad():
            if 'pixel_values' in inputs:
                _ = model(inputs['pixel_values'])
            else:
                _ = model(**inputs)
    except Exception as e:
        print(f"  ❌ Forward 실패: {e}")
    finally:
        for h in handles:
            h.remove()

    if captured_data:
        print(f"  ✅ {len(captured_data)}개 텐서 캡처")
        return [captured_data[-1]]
    
    return None

def overlay_attention(image, attn_map):
    if attn_map is None:
        return image

    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]
    
    mask = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.clip(mask, 0, 1)
    mask = mask ** 2 
    
    mask_3ch = np.stack([mask]*3, axis=-1)
    spotlight = img_np * (0.15 + 0.85 * mask_3ch) 
    
    return Image.fromarray(spotlight.astype(np.uint8))