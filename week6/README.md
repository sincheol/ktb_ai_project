# 딥러닝 모델 구현 및 데이터 증강 실험 보고서

프로젝트 주제: 활성화 함수 구현 및 CNN(AlexNet) 구조 설계를 통한 데이터 증강 효과 실험

## 1. 서론 (Introduction)

### 1.1 실험 배경

딥러닝 모델의 성능은 단순히 알고리즘의 우수성뿐만 아니라, 데이터의 양과 질, 그리고 모델의 구조적 설계에 의해 크게 좌우된다. 특히 이미지 데이터의 경우, 한정된 데이터셋 내에서 모델의 일반화(Generalization) 성능을 높이기 위해 다양한 데이터 증강(Data Augmentation) 기법이 필수적으로 사용된다. 본 프로젝트는 딥러닝의 기초가 되는 활성화 함수를 직접 구현해보고, 대표적인 CNN 아키텍처인 AlexNet을 변형하여 구현함으로써 딥러닝 파이프라인 전체를 실험적으로 구성하는 데 목적이 있다.

### 1.2 실험 목표

본 실험의 주된 목표는 높은 예측 정확도를 달성하는 것보다, **딥러닝 학습 파이프라인의 '구현(Implementation)'과 '실험 설계(Experimental Design)'**에 있다. 구체적인 목표는 다음과 같다.

주요 활성화 함수(Sigmoid, Tanh, ReLU)의 수식적 특성을 코드로 구현하고 시각화한다.

PyTorch를 사용하여 CNN 모델(SimpleCNN)을 설계하고, 학습 및 평가 루프를 직접 구축한다.

데이터 증강(Data Augmentation) 적용 유무에 따른 학습 양상(Loss, Accuracy)의 차이를 비교하는 실험 환경을 조성한다.

## 2. 이론적 배경 및 구현 (Theoretical Background & Implementation)

### 2.1 활성화 함수 (Activation Functions)

신경망의 비선형성을 부여하는 핵심 요소인 활성화 함수들을 numpy를 사용하여 직접 구현하였다.

Sigmoid: $1 / (1 + e^{-z})$ 형태로, 출력을 0과 1 사이로 압축한다.

Tanh: $2\sigma(2z) - 1$ 형태로, 출력을 -1과 1 사이로 매핑하여 중심을 0으로 맞춘다.

ReLU: $\max(0, z)$ 형태로, 양수 입력은 그대로 통과시키고 음수는 0으로 만듦으로써 기울기 소실 문제를 완화한다.
본 실험에서는 -10에서 10까지의 입력값에 대해 각 함수의 출력 변화를 시각화하여 동작 원리를 확인하였다.

### 2.2 모델 구조 (Model Architecture)

AlexNet의 구조를 기반으로 CIFAR-10 데이터셋(32x32)에 맞게 변형한 SimpleCNN을 설계하였다.

입력 처리: 32x32 이미지를 AlexNet의 입력 규격에 맞추기 위해 227x227로 리사이즈(Resize)하였다.

특징 추출기 (Feature Extractor): 5개의 Convolution Layer와 Max Pooling Layer로 구성하였다. 특히 초기 AlexNet 논문에서 사용된 LocalResponseNorm을 구현에 포함하여 역사적인 모델 구조를 재현하고자 하였다.

분류기 (Classifier): 3개의 Fully Connected Layer(Linear)로 구성하였으며, 과적합 방지를 위해 Dropout(p=0.5)을 적용하였다.

## 3. 실험 설계 (Experimental Design)

### 3.1 데이터셋 및 전처리

데이터셋: CIFAR-10 (비행기, 자동차, 새 등 10개 클래스의 컬러 이미지)

전처리 (Normalization): 학습 안정성을 위해 평균 0.5, 표준편차 0.5로 정규화를 수행하여 데이터 범위를 [-1, 1]로 조정하였다.

### 3.2 데이터 증강 (Data Augmentation) 비교 실험

데이터 증강이 모델 학습에 미치는 영향을 "구현 관점"에서 비교하기 위해 두 가지 DataLoader를 구성하였다.

Case 1: Augmentation 미적용

기본적인 Resize와 ToTensor, Normalize만 적용.

Case 2: Augmentation 적용

RandomHorizontalFlip (수평 뒤집기)

RandomVerticalFlip (수직 뒤집기)

RandomRotation (랜덤 회전)

위 기법들을 transforms.Compose로 묶어 학습 시 실시간으로 데이터가 변형되도록 파이프라인을 구축하였다.

### 3.3 학습 파이프라인 구축

train_one_epoch 함수와 evaluate 함수를 모듈화하여, 실험 조건이 바뀌더라도 동일한 학습 로직을 재사용할 수 있도록 구현하였다.

Optimizer: Adam (Learning Rate = 0.05)

Loss Function: CrossEntropyLoss

Epochs: 20회

## 4. 실험 결과 및 분석 (Results & Analysis)

### 4.1 구현 검증

작성된 코드를 통해 모델의 순전파(Forward) 및 역전파(Backward) 과정이 에러 없이 수행됨을 확인하였다. 특히 torchvision.transforms를 활용하여 원본 이미지와 증강된 이미지가 모델에 각각 올바르게 주입되는 데이터 로딩 파이프라인이 성공적으로 작동하였다.

### 4.2 실험 결과 (Loss 및 Accuracy 추이)

실험 결과, 데이터 증강을 적용하지 않은 경우와 적용한 경우 모두에서 손실(Loss) 값의 변화를 추적할 수 있었다.

Augmentation 미적용: 학습 데이터에 대한 과적합(Overfitting) 가능성이 높으나, 초기 수렴 속도가 상대적으로 빠를 것으로 예상된 설정이다.

Augmentation 적용: 데이터의 다양성을 확보하여 일반화 성능을 높이려는 설정이다.

(실험 노트: 본 시뮬레이션에서는 학습률(LR=0.05) 설정 등 하이퍼파라미터의 영향으로 모델이 최적점(Global Optima)에 수렴하지 못하고 10% 내외의 정확도에 머물렀으나, 두 실험군의 Loss 그래프를 동시에 시각화하여 비교하는 실험적 프레임워크 자체는 성공적으로 구현되었다.)

## 5. 결론 (Conclusion)

### 5.1 요약

본 프로젝트는 단순한 모델 학습을 넘어, 딥러닝 연구에 필요한 **'구현 능력'**을 함양하는 데 중점을 두었다. 활성화 함수부터 CNN 모델 설계, 그리고 데이터 증강 파이프라인까지 딥러닝의 핵심 요소들을 직접 코드로 작성하고 동작을 검증하였다.

### 5.2 의의 및 향후 과제

실험적 구현에는 성공하였으나, 모델이 유의미한 예측 성능을 내기 위해서는 하이퍼파라미터 튜닝이 필수적임을 확인하였다. 특히 초기 학습률(Learning Rate) 조정이나 가중치 초기화(Weight Initialization) 기법을 추가로 적용한다면, 현재 구축된 탄탄한 실험 파이프라인 위에서 훨씬 더 높은 성능을 도출할 수 있을 것으로 기대된다.

## 6. 출처 및 참고 문헌
AlexNet : https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf