# Transformer 모델 구현 및 기계 번역 실험 보고서

프로젝트 주제: "Attention Is All You Need" 논문 기반 Transformer 아키텍처 구현 및 독일어-영어 번역 실험


## 1. 서론 (Introduction)

### 1.1 실험 배경

현대 자연어 처리(NLP)의 기반이 되는 Transformer 모델은 RNN이나 CNN 없이 오직 어텐션(Attention) 메커니즘만으로 시퀀스 데이터를 처리하는 혁신적인 구조이다. 본 프로젝트는 라이브러리에서 제공하는 완제품 모델을 사용하는 대신, 논문에 기술된 수식을 PyTorch 코드로 직접 구현함으로써 모델의 내부 동작 원리를 깊이 있게 이해하고, 실제 번역 태스크(Multi30k)에 적용하여 그 유효성을 실험적으로 검증하는 데 목적이 있다.

### 1.2 실험 목표

본 프로젝트는 SOTA(State-of-the-Art) 성능 달성보다는 복잡한 딥러닝 아키텍처의 단계별 구현 및 실험적 검증에 초점을 맞추었다.

핵심 모듈 구현: Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding 등 Transformer의 핵심 구성 요소를 밑바닥부터 구현한다.

아키텍처 조립: 인코더(Encoder)와 디코더(Decoder) 스택을 구성하고, 마스킹(Masking) 기법을 통해 학습 시 미래 정보 참조를 방지하는 로직을 구축한다.

번역 파이프라인 구축: 데이터 전처리부터 모델 학습, 추론(Greedy Decoding), BLEU 점수 평가까지 전체 파이프라인을 실험적으로 구성한다.

## 2. 모델 아키텍처 구현 (Model Architecture Implementation)

본 실험에서는 Transformer의 전체 구조를 모듈화하여 클래스로 구현하였다.

### 2.1 어텐션 메커니즘 (Attention Mechanism)

Scaled Dot-Product Attention: 쿼리(Q), 키(K), 밸류(V)를 이용한 어텐션 연산을 직접 구현하였다. 특히 $ \sqrt{d_k} $로 스케일링하여 기울기 소실 문제를 방지하는 수식을 코드로 반영하였다.

Multi-Head Attention: 입력 벡터를 여러 헤드(Head)로 분할하여 병렬 처리하고 다시 병합(Concat)하는 과정을 구현하여, 모델이 다양한 문맥 정보를 동시에 포착할 수 있도록 설계하였다.

### 2.2 위치 인코딩 (Positional Encoding)

순서 정보가 없는 어텐션 메커니즘의 한계를 극복하기 위해, 사인(Sine)과 코사인(Cosine) 함수를 이용한 위치 인코딩을 구현하여 입력 임베딩에 더해주었다.

### 2.3 인코더-디코더 구조

Encoder: Self-Attention과 Feed-Forward Network(FFN)로 구성된 레이어를 $N=3$회 쌓아 입력 문장의 문맥을 추출하도록 구현하였다.

Decoder: Masked Self-Attention을 통해 자기 회귀(Auto-regressive) 속성을 유지하며, 인코더의 출력과 Cross-Attention을 수행하여 번역문을 생성하도록 설계하였다.

Masking: 패딩(Padding) 토큰을 무시하기 위한 마스크와, 디코더가 미래의 단어를 미리 보지 못하게 하는 Look-ahead Mask를 논리적으로 구현하였다.

## 3. 실험 설계 및 환경 (Experimental Setup)

### 3.1 데이터셋 및 전처리

Dataset: Multi30k (독일어 $\rightarrow$ 영어 번역 데이터셋). torchdata 버전 호환성 이슈를 해결하기 위해 Hugging Face의 datasets 라이브러리를 보조적으로 활용하는 실험적 구성을 취하였다.

Tokenization: spaCy 라이브러리의 독일어/영어 모델을 사용하여 토큰화를 수행하였다.

Vocabulary: 학습 데이터에서 최소 빈도수 2 이상인 단어들로 어휘집을 구축하였다.

### 3.2 학습 설정 (Training Configuration)

Hyperparameters:

Embedding Dimension ($d_{model}$): 256

Feed-forward Dimension ($d_{ff}$): 512

Attention Heads ($n_{head}$): 8

Layers ($N$): 3

Dropout: 0.1

Optimization: AdamW 옵티마이저(LR=0.0005)를 사용하고, CrossEntropyLoss로 손실을 계산하였다. 안정적인 학습을 위해 Gradient Clipping(max_norm=1.0)을 적용하였다.

## 4. 실험 결과 및 분석 (Results & Analysis)

### 4.1 학습 진행 (Training Progress)

총 10 Epoch 동안 학습을 진행하였으며, 학습이 진행됨에 따라 Loss가 안정적으로 감소함을 확인하였다.

초기 Loss: ~4.09

최종 Loss: ~1.27 (10 Epoch 기준)
이는 직접 구현한 어텐션 연산과 역전파 과정이 수학적으로 올바르게 동작하고 있음을 시사한다.

### 4.2 정량적 평가 (BLEU Score)

학습된 모델의 성능을 기계 번역 표준 지표인 BLEU Score로 평가하였다.

평가 방식: torchtext.data.metrics.bleu_score를 활용하여 테스트 데이터셋에 대한 예측 문장과 정답 문장의 n-gram 일치도를 계산하였다.

최종 결과: Test BLEU Score 38.82

이는 학습 데이터가 상대적으로 적은 Multi30k 환경에서도 Transformer 구조가 효과적으로 작동함을 보여준다.

### 4.3 정성적 평가 (Inference)

translate_sentence 함수를 통해 독일어 문장을 입력받아 영어로 번역하는 과정을 테스트하였다. Greedy Decoding 방식을 사용하여 매 시점 가장 확률이 높은 단어를 선택하며 문장을 생성하는 과정을 검증하였다.

## 5. 결론 (Conclusion)

### 5.1 요약

본 프로젝트는 Transformer 아키텍처의 End-to-End 구현을 통해 딥러닝 모델링 능력을 함양하는 데 중점을 두었다. 논문의 수식을 코드로 옮기는 과정에서 마스킹, 차원 변환, 어텐션 스코어 계산 등의 디테일을 실험적으로 검증하였다.

### 5.2 의의 및 향후 과제

직접 구현한 모델이 38.82라는 준수한 BLEU 점수를 기록함으로써 구현의 정확성을 입증하였다. 향후에는 Greedy Decoding 대신 Beam Search를 구현하거나, 학습률 스케줄링(Warm-up) 기법을 추가 적용하여 성능을 더욱 고도화할 수 있는 실험적 기반을 마련하였다.

## 6. 출처 및 참고 문헌
Attention Is All You Need : https://arxiv.org/pdf/1706.03762