# ViT
A project to compare the visual focus of a custom fine-tuned ViT model and a SOTA InternViT model using attention maps on the Bean Leaf dataset.

## 부제: ViT와 InternViT 비교로 기술 변화 체감하기

## 파일 구조 설명
```
ViT/
├── outputs/                    # 학습 결과물 저장소
│   ├── vit-base-beans/         # 체크포인트 및 모델 저장 경로
│   ├── confusion_matrix.png    # CM
│   ├── final_metrics.png       # evaluation 결과
│   └── history_table.png       # 학습 중 epoch별 loss 및 성능 변화
├── src/                        # 핵심 소스 코드 패키지
│   ├── __pycache__             # 파이썬 컴파일 캐시
│   ├── __init__.py             # 패키지 초기화 파일
│   ├── dataset.py              # 데이터셋 로드 및 전처리
│   ├── model.py                # ViTForImageClassification 모델 호출 및 설정
│   ├── trainer_utils.py        # evaluation 함수 모음
│   └── config.py               # hyperparameter 설정
├── train.py                    # train, eval, 시각화
├── inference.py                # 단일 이미지 예측용 스크립트 (CLI 환경 테스트용)
├── requirements.txt            # 프로젝트 실행에 필요한 라이브러리 목록
├── app.py                      # Streamlit 웹 대시보드 (Base ViT vs Fine-tuned ViT vs InternViT 비교)
├── vis_utils.py                # Attention Map 추출 및 시각화(히트맵 오버레이) 함수
└── README.md                   # 프로젝트 설명
```

## 코드 실행 과정

- 1.무슨 환경 있나 확인
    - `conda env list` 

- 2. 만일 환경이 없다면 (만일 이전 명령어에서 fine_tune_vit라는 환경이 있으면 이거 스킵)
    - `conda create --name fine_tune_vit python=3.11`

- 3. 가상환경 켜기
    - `conda activate fine_tune_vit` 

- 4. 파일 경로 이동
    - `cd ViT`

- 5. 필수 라이브러리 설치 (최초 1회만)
    - `pip install -r requirements.txt`

- 6. InternViT 위해
    - `pip install flash_attn --no-build-isolation`

- 7. GPU 켜고 코드 실행
    - `CUDA_VISIBLE_DEVICES=0 python train.py`

- 8. 학습 끝나면
    - `streamlit run app.py`
