# 전이 학습(Transfer Learning) 및 하이퍼파라미터 최적화 파이프라인 구축 실험 보고서

프로젝트 주제: ResNet50과 VGG16을 활용한 Food-101 분류 및 실험적 튜닝 프레임워크 구현

## 1. 서론 (Introduction)

### 1.1 실험 배경

대규모 데이터셋인 Food-101(약 5GB, 101개 클래스)을 효율적으로 학습시키기 위해서는 검증된 아키텍처를 활용한 전이 학습(Transfer Learning)이 필수적이다. 본 프로젝트는 단순히 사전 학습된 모델을 가져와 사용하는 것에 그치지 않고, 특징 추출(Feature Extraction)에서 미세 조정(Fine-tuning)으로 이어지는 학습 전략을 단계별로 구현하고, 최적의 성능을 도출하기 위한 실험적 파이프라인(HPO, Logging)을 직접 구축하는 데 목적을 둔다.

### 1.2 실험 목표

본 프로젝트의 핵심은 모델의 최종 정확도보다는 다음과 같은 실험적 구현 능력을 입증하는 데 있다.

전이 학습 전략 구현: 모델의 레이어를 선택적으로 동결(Freeze) 및 해제(Unfreeze)하는 코드를 작성하여 학습 전략을 제어한다.

아키텍처 비교 분석: 구조가 상이한 ResNet50과 VGG16의 학습 양상을 실험적으로 비교하고, 구조적 차이(Batch Norm, Skip Connection)가 학습 안정성에 미치는 영향을 분석한다.

최적화 파이프라인 구축: Grid Search와 Random Search 알고리즘을 코드로 구현하여 최적의 하이퍼파라미터를 탐색한다.

MLOps 도구 활용: Weights & Biases(W&B)를 연동하여 실험 결과를 체계적으로 로깅하고 시각화한다.

## 2. 실험 환경 및 데이터 처리 (Experimental Setup)

### 2.1 데이터셋 파이프라인

Dataset: Food-101 (101,000 images, 101 classes).

Preprocessing: 대용량 이미지 데이터를 효율적으로 처리하기 위해 ImageNet 기준의 정규화(Mean/Std)와 리사이즈(224x224)를 적용하였다.

Data Augmentation: 학습 데이터에 RandomResizedCrop, RandomHorizontalFlip을 적용하여 모델의 일반화 성능을 높이고 과적합을 방지하는 파이프라인을 구성하였다.

Efficiency: DataLoader에서 num_workers=4, pin_memory=True 옵션을 사용하여 GPU 활용 효율을 극대화하였다.

## 3. 모델 구현 및 학습 전략 (Implementation Strategies)

### 3.1 모델 아키텍처 및 전이 학습 설계

두 가지 대표적인 CNN 백본을 사용하여 실험을 설계하였다.

ResNet50: 잔차 연결(Residual Connection)과 배치 정규화(Batch Normalization)가 포함된 깊은 망.

VGG16: 깊지만 단순한 구조를 가지며 파라미터 수가 매우 많은 망.

[구현 전략 1: Feature Extraction]

모든 사전 학습된 파라미터의 requires_grad를 False로 설정하여 동결(Freeze)하였다.

마지막 분류기(Classifier/FC Layer)만 교체하여 해당 부분만 학습되도록 코드를 구현하였다.

[구현 전략 2: Fine-Tuning (Differential Learning Rates)]

모델의 마지막 Convolution Block(예: ResNet의 layer4, VGG의 상위 features)을 동결 해제(Unfreeze)하였다.

차등 학습률(Differential Learning Rate) 적용: 이미 잘 학습된 Backbone에는 매우 낮은 학습률(예: 1e-5)을, 새로 추가된 Head에는 상대적으로 높은 학습률(예: 1e-3)을 부여하는 Optimizer 그룹을 코드로 구현하여 기존 지식의 파괴를 최소화하며 학습시켰다.

## 4. 실험적 관찰 및 분석 (Experimental Observations)

### 4.1 모델 구조에 따른 학습 안정성 비교

본 실험에서는 동일한 학습률(Learning Rate = 0.005) 환경에서 두 모델의 초기 학습 양상을 비교 분석하였다.

관찰: ResNet50은 안정적으로 수렴하였으나, VGG16은 초기 Loss가 발산하거나 수렴하지 못하는 현상이 발생하였다.

분석:

Batch Normalization: ResNet은 BN 층이 있어 높은 학습률에서도 가중치 분포가 안정적이지만, VGG는 BN이 없어 초기 학습률에 매우 민감하다.

Classifier 구조: VGG16의 분류기(FC layer)는 전체 파라미터의 약 80%를 차지할 정도로 무거워, 높은 학습률이 전체 모델을 크게 흔들 수 있음을 확인하였다.

해결: VGG16의 학습률을 0.0001로 낮추어 안정적인 학습을 유도하는 실험적 튜닝을 수행하였다.

### 4.2 하이퍼파라미터 최적화 (HPO) 구현

단순한 매뉴얼 튜닝을 넘어, 자동화된 탐색 알고리즘을 코드로 구현하였다.

Grid Search: 학습률([0.01, ..., 0.000005])과 옵티마이저(Adam, SGD)의 모든 조합을 탐색하는 루프를 작성하였다.

Random Search: 탐색 범위를 지정하고 로그 스케일(10 ** random.uniform)로 학습률을 무작위 샘플링하여, 제한된 자원 내에서 더 넓은 공간을 효율적으로 탐색하는 로직을 구현하였다.

실험 관리: wandb.init()과 wandb.log()를 루프 내에 배치하여 각 실험(Trial)의 설정과 결과가 자동으로 기록되고 비교되도록 시스템을 구축하였다.

## 5. 결론 (Conclusion)

### 5.1 요약

본 프로젝트는 Food-101 데이터셋을 대상으로 딥러닝 실험 환경의 End-to-End 구현을 성공적으로 수행하였다. 특히 전이 학습의 단계별 적용(Freeze -> Unfreeze), 모델 구조에 따른 트러블슈팅, 그리고 자동화된 하이퍼파라미터 튜닝 시스템 구축을 통해 모델 개발 프로세스 전반에 대한 기술적 이해도를 높였다.

### 5.2 의의

단순히 "정확도 몇 %를 달성했다"는 결과보다, **"왜 VGG는 ResNet보다 학습률에 민감한가?"**와 같은 질문을 던지고 이를 코드로 검증해내는 과정에 집중하였다. 이러한 실험적 접근과 MLOps 도구(W&B)의 활용 경험은 향후 더 복잡한 딥러닝 문제를 해결하는 데 있어 견고한 기반이 될 것이다.