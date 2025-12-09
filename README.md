# 인공지능 프로젝트 종합 요약 보고서

## 1. 개요 (Overview)

본 보고서는 딥러닝의 기초부터 최신 모델 구현까지 단계별로 수행한 4가지 인공지능 프로젝트의 주요 내용과 실험 결과를 요약한다. 각 프로젝트는 데이터 분석, 모델 설계, 성능 최적화, 그리고 논문 구현 능력을 함양하는 것을 목표로 진행되었다.

## 2. 프로젝트별 요약 (Project Summaries)

### Project 1: 뇌졸중 발병 예측 (Stroke Prediction)

* 목표: 인구통계학적 정보와 건강 데이터를 기반으로 뇌졸중 발병 여부를 이진 분류(0/1)하는 최적의 머신러닝 모델 탐색.

* 모델: KNN, Perceptron, Linear SVM, Random Forest, Naive Bayes.

* 주요 실험: 데이터 불균형(Class Imbalance) 문제 해결을 위해 다양한 알고리즘을 비교하고, 정확도(Accuracy)뿐만 아니라 재현율(Recall)을 중점적으로 평가.

* 성과: 전처리 파이프라인 구축 및 머신러닝 모델의 성능 비교 분석 수행. 데이터 불균형이 모델 성능에 미치는 영향을 실험적으로 확인.

### Project 2: 활성화 함수 및 CNN 기초 구현 (Activation & CNN Implementation)

* 목표: 딥러닝의 핵심 요소인 활성화 함수와 CNN 아키텍처를 직접 코드로 구현하여 동작 원리를 이해.

* 모델: Sigmoid, Tanh, ReLU (활성화 함수), SimpleCNN (AlexNet 변형).

* 주요 실험: 활성화 함수의 수식적 구현 및 시각화, torchvision을 활용한 데이터 증강(Data Augmentation) 파이프라인 구축, 증강 유무에 따른 학습 양상 비교.

* 성과: 딥러닝 학습 루프(Train/Eval)의 밑바닥 구현 및 데이터 증강 효과 검증.

### Project 3: 전이 학습 및 HPO 파이프라인 (Transfer Learning & HPO)

* 목표: 대규모 데이터셋(Food-101) 처리를 위한 전이 학습 전략 수립 및 하이퍼파라미터 최적화(HPO) 시스템 구축.

* 모델: ResNet50, VGG16 (Pre-trained).

* 주요 실험: Feature Extraction vs Fine-tuning 비교, 차등 학습률(Differential Learning Rate) 적용, Grid Search 및 Random Search 구현, W&B를 이용한 실험 로깅.

* 성과: 모델 구조(ResNet vs VGG)에 따른 학습 안정성 차이 분석 및 자동화된 튜닝 파이프라인 확보.

### Project 4: Transformer 기계 번역 (Transformer Translation)

* 목표: "Attention Is All You Need" 논문의 Transformer 모델을 밑바닥부터 구현하여 기계 번역 태스크 수행.

* 모델: Transformer (Encoder-Decoder Architecture).

* 주요 실험: Multi-Head Attention, Positional Encoding, Masking 로직 직접 구현, 독일어-영어 번역 모델 학습 및 BLEU Score 평가.

* 성과: 복잡한 Transformer 아키텍처의 수식적 구현 검증 및 BLEU 38.82 달성. Hugging Face 라이브러리를 활용한 데이터 파이프라인 트러블슈팅.

## 3. 종합 결론 (Conclusion)

일련의 프로젝트를 통해 정형 데이터 분석부터 컴퓨터 비전(CNN), 자연어 처리(Transformer)에 이르는 다양한 도메인의 딥러닝 기술을 습득하였다.

특히 단순한 라이브러리 활용을 넘어, 모델의 구조를 직접 설계하고 구현하는 능력과 실험을 통해 성능을 분석하고 개선하는 역량을 확보하였다는 점에서 의의가 있었다.