# HarmonyRL — 연구 기록

[English](README.md) | 한국어

강화학습을 이용한 멜로디 하모나이제이션입니다. 멜로디가 주어지면 모델이 (코드, 길이) 이벤트의
시퀀스로 코드 진행을 씁니다. Transformer 정책을 사람 반주로 사전학습한 뒤, KL로 정규화한
강화학습으로 미세조정합니다.

이 저장소에는 연구 코드(데이터 준비, 학습, 평가, 청취 도구)가 있습니다. 여기서 나온 모델은
따로 배포합니다.

- 라이브러리: [`HarmonyRL/HarmonyRL`](https://github.com/HarmonyRL/HarmonyRL), PyPI
  [`harmonyrl`](https://pypi.org/project/harmonyrl/)
- 모델: [`CloudHolic/harmonyrl-chromatic`](https://huggingface.co/CloudHolic/harmonyrl-chromatic),
  [`-smooth`](https://huggingface.co/CloudHolic/harmonyrl-smooth),
  [`-base`](https://huggingface.co/CloudHolic/harmonyrl-base),
  [HarmonyRL 컬렉션](https://huggingface.co/collections/CloudHolic/harmonyrl)

**요약.** 사람 데이터나 청취자 선호로 학습한 보상은 청취자가 더 좋아하는 출력을 고르지
못했습니다. 그래서 최종 보상은 제품 목표로 정했습니다. 사전학습 정책에 대한 KL 페널티가 허용하는
한 조성 밖 코드를 최대한 쓰는 것입니다. 이 보상으로 RL을 하자 조성 밖 코드가 울리는 시간이 두
배가 됐고(TEST 6.9% → 12.9%), 블라인드 청취에서는 다시 뽑을 수 있을 때 사전학습 정책과 같은
수준으로 평가됐습니다. 아래 두 가설 모두 귀로는 차이가 없었습니다.

## 목차

1. [과제](#과제)
2. [가설](#가설)
3. [데이터](#데이터)
4. [방법](#방법)
5. [효과가 없었던 보상](#효과가-없었던-보상)
6. [평가](#평가)
7. [결과](#결과)
8. [배포 모델](#배포-모델)
9. [재현](#재현)
10. [저장소 구조](#저장소-구조)
11. [한계](#한계)
12. [라이선스와 참고 문헌](#라이선스와-참고-문헌)

## 과제

**입력.** 멜로디 구간: 8분음표 격자 위의 음표, 박자표(구간 안에서 바뀌지 않는 `x/4`·`x/8`이면
무엇이든), 시간에 따른 국소 조성, 그리고 있으면 섹션 이름과 템포.

**출력.** 구간을 빈틈없이 덮는 코드 이벤트 시퀀스. 이벤트 하나는 (코드, 길이)이고 길이는
8분음표 단위(1–128)입니다. 에피소드 하나가 구간 하나이며, 에피소드 길이는 코드 개수입니다.

**코드 표현.** 코드는 Hooktheory 데이터와 같이 국소 조성에 대한 상대 표기로 쓰고, 성분으로
나눠 모델이 차례로 예측합니다.

| 헤드 | 값 |
|---|---|
| root | 음계 도수 1–7, 또는 무코드 |
| applied | 0–7 (부속 기능, 예: V/V) |
| borrowed | 차용한 선법, 또는 없음 (선법 9개) |
| type | 3화음, 7, 9, 11, 13 |
| suspensions | 없음, sus2, sus4, 둘 다 |
| adds | add 4, 6, 9의 조합 |
| omits | 없음, 3음 생략, 5음 생략, 둘 다 |
| alterations | ♭5, ♯5, ♭9, ♯9, ♯11, ♭13의 조합 |
| inversion | 0–3 |
| duration | 8분음표 1–128개 |

**무코드.** 생성한 하모나이제이션은 첫 온전한 마디 앞의 못갖춘마디 밑에서만 코드를 비울 수
있습니다. 그 밖에서는 학습과 생성 모두 "무코드"를 마스킹으로 막습니다.

## 가설

목표는 주어진 멜로디에 맞는 코드 진행을 잘 생성하는 모델입니다. 그 과정에서 두 가설을
검증했고, 실험 순서와 실행 횟수는 가설이 아니라 목표에 맞춰 정했습니다.

| | 질문 | 답 |
|---|---|---|
| **H1** | 시퀀스 단위 보상이 음표 단위 국소 보상(RL-Chord 방식)보다 더 자연스러운 진행을 만드는가? | **지지되지 않음.** 6회차에서 국소 보상 모델이 20곡 중 12곡에서 선호됐고, 7회차에서는 4 : 6으로 갈렸습니다. 둘 다 유의하지 않습니다. |
| **H2** | 같은 목표 분포 π_ref · exp(R/β)에 대해, 순방향 KL 학습(DPG)이 역방향 KL 학습(RLOO)보다 다양성을 더 잘 보존하는가? | **의미 있는 차이 없음.** TEST 샘플 간 불일치 0.808(순방향) 대 0.800(역방향), 조성 밖 비율은 같음(12.9%), 청취 10곡 중 4곡. |

H1의 시퀀스 단위 보상은 결국 학습된 평가자가 아니라 [방법](#방법)에서 설명하는 제품 목표가
됐습니다. 학습된 보상 중 청취 확인을 통과한 것이 없었기 때문입니다
([효과가 없었던 보상](#효과가-없었던-보상)).

## 데이터

[Sheet Sage](https://github.com/chrisdonahue/sheetsage)와 함께 공개된
[Hooktheory](https://www.hooktheory.com)의 멜로디와 코드
([데이터 저장소](https://github.com/chrisdonahue/sheetsage-data), 커밋 `06113c0`). 단순화
파일(`Hooktheory.json.gz`)과 원본 파일(`Hooktheory_Raw.json.gz`)을 함께 읽습니다. 원본 파일에서
전체 표기의 코드, 섹션 이름, 템포를 가져옵니다.

| 단계 | 항목 수 |
|---|---|
| 데이터셋의 항목 | 26,175 |
| 멜로디와 코드가 모두 있음 | 23,612 |
| 항목 안에서 박자표가 바뀜 (제외) | 269 |
| 표현할 수 없는 코드 (제외) | 2 |
| **사용한 구간** | **23,341** |
| — 그중 구간 안에서 전조 | 473 |

구간 안에서 바뀌지 않는 한 모든 박자표를 씁니다. 4/4 21,195, 6/4 1,127, 3/4 843, 6/8 47,
2/4 44, 12/8 32, 5/4 25, 그 밖 30. 전조도 그대로 두고, 국소 조성을 모델 입력으로 줍니다.

분할은 데이터셋 자체의 `split` 필드를 씁니다.

| TRAIN | VALID | TEST |
|---|---|---|
| 18,937 | 1,931 | 2,473 |

모든 선택은 VALID에서 했고, TEST는 마지막에 한 번만 평가했습니다.

## 방법

### 사전학습 정책 π_ref

`ChordPolicy`는 인코더–디코더 Transformer입니다(파라미터 18.4M).

- **인코더**는 멜로디를 8분음표 하나당 프레임 하나로 읽습니다. 프레임 입력: 그 프레임에서
  울리는 음높이 클래스(국소 으뜸음 기준, 겹치는 길이로 가중), 음 시작, 옥타브, 마디 안 위치,
  박의 세기, 국소 선법, 조성 변화. 구간마다: 마디당 박 수, 박 단위, 섹션 이름, 템포 구간.
- **디코더**는 지금까지의 코드 이벤트를 읽습니다. 이벤트마다 헤드를 하나씩 예측하며, 각 헤드는
  앞서 고른 헤드에 조건부이고, 행동 마스크가 코드를 유효하게 유지합니다.
- 384차원, 어텐션 헤드 4개, 인코더 4층·디코더 4층, dropout 0.3.
- TRAIN에서 최대우도로 학습: AdamW, 학습률 3e-4(warmup 500 step), weight decay 0.1, 최대
  15 epoch에 조기 종료. 섹션 이름과 템포는 각각 20% 확률로 지워서 없어도 동작하게 했습니다.
- 최고 epoch 13: VALID 이벤트당 음의 로그우도 2.529. 헤드별 정확도: root 0.719, type 0.899,
  inversion 0.909, duration 0.807.

### 기준선

- **HMM**, Yeh et al.(2021)을 따름: 강박에서 다음 강박까지 슬롯마다 코드 하나, Viterbi 디코딩.
- **π_ref 탐욕 디코딩**과 **온도 0.85의 π_ref 샘플링**. T 1로 샘플링하면 목적 없이 어긋나는
  코드가 나왔고, 탐욕 디코딩은 조성 밖 코드를 대부분 없앴습니다(시간 비율 2.9%, 사람은 9.6%).
  T 0.85는 청취로 고른, RL이 이겨야 할 디코딩입니다. RL 없이도 T 1의 π_ref는 짧은 곡의 블라인드
  확인에서 이미 사람 반주와 구분되지 않았습니다(20곡 중 10곡).

### 강화학습

두 KL 방향 모두 같은 목표 분포로 미세조정합니다.

&nbsp;&nbsp;&nbsp;&nbsp;p(y) ∝ π_ref^(1/0.85)(y) · exp(R(y) / β)

π_ref^(1/0.85)는 사전학습 정책을 온도 0.85로 날카롭게 한 것으로, 청취자가 안전한 기준으로 들은
디코딩입니다. 정책도 같은 날카롭게 한 가중치에서 시작하므로 KL은 0에서 출발하고, 학습 후에는
T 1로 디코딩합니다.

멜로디마다 현재 정책에서 샘플 k = 8개를 뽑고, 각 샘플에 로그 비 s = R/β + log π_ref(y) − log π(y)를
계산합니다. 두 방법은 rollout, 옵티마이저, 기록을 공유하고, −Σ w · log π(y)에서 샘플에 주는
가중치만 다릅니다.

| | 발산 | 가중치 w |
|---|---|---|
| **RLOO** (Ahmadian et al., 2024) | 역방향 KL, KL(π ‖ p) | β · (s − 같은 멜로디의 다른 샘플들의 평균 s) |
| **DPG** (Parshakova et al., 2019; Khalifa et al., 2021) | 순방향 KL, KL(p ‖ π) | 같은 멜로디의 샘플들에 대한 k · softmax(s) |

가치 함수와 PPO는 쓰지 않습니다. 모든 실행의 설정: TRAIN에서 멜로디 32개 × 샘플 8개로 200
update, 학습률 1e-5, gradient clip 1.0, seed 0.

### 최종 보상

보상은 에피소드 끝에 한 번 줍니다.

&nbsp;&nbsp;&nbsp;&nbsp;R = (조성 밖 코드 아래의 시간 비율) − 2 × (코드 하나가 2마디를 넘게 지속된 시간 비율)

- **조성 밖**: 코드에 국소 음계 밖의 음높이 클래스가 있음. 단조는 자연·화성·선율단음계의
  합집합을 음계로 보고, 그 밖의 선법은 각자의 음계를 씁니다. 이 항은 코드가 맞는지를 판단하는
  것이 아니라 제품이 원하는 것(풍부한 리하모니제이션)을 나타냅니다. 자리마다 얼마나 나갈지는
  KL 항을 통해 π_ref가 정하고, 전체 과감함은 청취로 고른 β(0.2)가 정합니다.
- **지속**: 같은 음높이 클래스(전위 포함)가 2마디를 넘게 울림. 청취자는 긴 지속을 싫어했고(두
  하모나이제이션 중 한쪽만 코드를 3마디 이상 지속하면 다른 쪽이 42번 중 33번 이김), π_ref는
  데이터만큼 지속하므로 KL 항으로는 막히지 않습니다.

화성을 판단하는 그 밖의 규칙은 보상에 넣지 않았습니다.

### 국소 보상 기준선 (H1)

RL-Chord 보상(Ji et al., 2024)의 핵심을 다시 구현한 것으로, 멜로디의 음표와 쉼표에 대해
평균합니다.

- 같은 시점의 사람 코드와의 일치(공유하는 음높이 클래스에서 베이스·음높이 클래스·무코드 불일치를
  뺌);
- 코드와 주변 멜로디 음의 협화도, 코드 자체의 협화도;
- 진행: 마디선을 넘어 지속하면 −1, 마디 안에서 지속하면 +1, 앞 코드와 공통음이 없으면 −1.

보상의 척도가 달라 β를 따로 골랐고, RLOO로 β 0.8에서 학습했습니다.

## 효과가 없었던 보상

프로젝트의 대부분은 청취자가 좋아하는 출력을 고르는 보상을 찾는 데 들었습니다. 모든 후보는
먼저 RL 없이 확인했습니다. 보상이 N개 샘플 중에서 고른 출력(best-of-N)이 더 좋게 들리지 않으면,
그 보상으로 RL을 해도 나아질 이유가 없기 때문입니다. 아래 확인은 모두 1인 청취자의 T 0.85 π_ref
대비 블라인드 청취 비교입니다.

| 보상 | 학습 방식 | best-of-N 대 T 0.85 | 학습에 안 쓴 판정과의 일치 |
|---|---|---|---|
| 처음부터 학습한 판별기 | 프레임 Transformer, 사람 대 π_ref 샘플 | best-of-8 11 : 10, best-of-32 10 : 12 | 우연 수준 |
| MusicBERT 미세조정 판별기 | 사람 대 샘플 AUC 0.766 | best-of-32 8 : 14 | 우연 수준 |
| 선호 보상, MusicBERT 특징 | Bradley–Terry, 고정 특징 위 선형 헤드 | — | 0.44–0.56 |
| 선호 보상, π_ref 상태 | 디코더 상태 위 Bradley–Terry | best-of-32 16 : 16 | 교차검증 0.64 → 새 곡 0.49 |
| 선호 보상, log π_ref | 이벤트 로그우도의 평균·최소 위 Bradley–Terry | best-of-32 14 : 18 | 0.59–0.60 → 새 곡 0.54 |

선호 보상은 소유자의 쌍·순위 판정 348개로 학습하고 확인했습니다. 기호 입력 LLM 판정자, 규칙
감점의 합, P(멜로디 | 코드)도 시도했지만 모두 우연 수준이었습니다.

알게 된 것:

- **판정 자체는 일관됐습니다**(반복 비교 18번 중 17번 일치). 신호는 귀에 있었고, 특징이 그것을
  담지 못했습니다.
- **수백 개 판정의 교차검증은 낙관적이었습니다.** 0.69와 0.64의 정확도가 새 곡에서 사라졌습니다.
  보상은 학습에 쓰지 않은 곡의 청취가 맞을 때만 믿었습니다.
- **고차원 특징은 수백 개 판정에 과적합합니다.** 작은 특징은 π_ref 로그우도 이상의 정보가 없었고,
  그 신호는 금방 포화합니다.
- **선호는 가능도에 단조롭지 않습니다.** 비슷한 후보 사이에서는 더 그럴듯한 쪽이 이기지만(0.59),
  탐욕 디코딩은 샘플링을 이기지 못합니다. π_ref는 "과감해서 드문 것"과 "틀려서 드문 것"을 가르지
  못합니다.
- **"사람다움"은 잘못된 목표입니다.** 데이터는 모든 장르 곡을 사용자가 받아 적은 것으로, 의도된
  기묘함과 받아 적기 잡음이 섞여 있고, 판별기는 그것까지 배웁니다.
- **모델의 탐욕 방향은 청취자의 방향과 반대입니다.** 그럴듯한 코드는 흔한 코드이고, 그러면
  청취자가 어울릴 때 높이 평가한 조성 밖 코드가 사라집니다.

그래서 최종 보상에 이르렀습니다. 판정을 예측하는 대신 목표(안전한 범위에서 조성 밖 코드를
최대한)를 정하고, 자리마다 무엇이 안전한지는 KL 항을 통해 π_ref가 정하게 했습니다.

## 평가

### 지표

`scripts/eval/evaluate.py`로 모든 시스템을 같은 구간에서 사람 반주와 비교해 채점했습니다(정책은
구간당 샘플 8개).

- **Yeh et al.(2021)**: chord histogram entropy(CHE), chord coverage(CC), chord tonal
  distance(CTD), chord-tone to non-chord-tone ratio(CTnCTR), pitch consistency score(PCS),
  melody–chord tonal distance(MCTD).
- **구조**: 반복 프레이즈에서 코드의 일치, 조성과의 일관성, 프레이즈 끝과 안쪽의 코드 분포(사람과의
  Jensen–Shannon 거리).
- **어법**: 강박의 비화성음과 그 해결, 텐션 비율, 차용 화음, 부속 화음.
- **다양성**: 사람과의 코드 2·3-gram 거리, 같은 멜로디의 샘플 간 불일치, TRAIN에서 그대로 가져온
  구간의 길이.
- **보상 항**: 조성 밖 비율과 지속 비율.

최종 보상의 두 항을 빼면 어떤 지표도 보상으로 쓰지 않았습니다.

### 청취

결정은 1인 청취자인 프로젝트 소유자의 블라인드 청취로 했습니다. 세트는 작게(VALID 10–20곡) 여러
회차에 걸쳐 했고, 순서를 무작위로 섞고 2그룹을 반복해 자기 일관성을 확인했습니다. 뒤 회차에서는
곡마다 후보 K개에 순위를 매기고, 각각 수용 가능 여부와 문제가 있는 마디를 표시했습니다. 청취용으로만
각 세트를 첫 조성이 흰건반 조(장조 C, 단조 A)가 되도록 조옮김했고, 모델은 이를 보지 않습니다. 판정은
공개하지 않습니다.

## 결과

### TEST (구간 2,473개)

| | 사람 | HMM | π_ref 탐욕 | π_ref T 0.85 | RL 역방향 | RL 순방향 | RL 국소 |
|---|---|---|---|---|---|---|---|
| 조성 밖 비율 | 9.6% | 5.8% | 2.9% | 6.9% | **12.9%** | **12.9%** | 5.5% |
| 지속 비율 (2마디 초과) | 3.2% | 16.0% | 7.4% | 4.1% | 1.4% | 2.1% | 5.3% |
| 차용 화음 | 11.0% | 6.6% | 4.7% | 8.4% | 13.1% | 13.4% | 7.2% |
| 부속 화음 | 4.2% | 0.5% | 0.6% | 3.0% | 4.5% | 4.5% | 2.2% |
| 마디당 코드 수 | 1.34 | 0.88 | 1.13 | 1.29 | 1.32 | 1.31 | 1.20 |
| 샘플 간 불일치 | – | – | – | 0.767 | 0.800 | 0.808 | 0.716 |
| 사람과의 코드 3-gram JS | 0 | 0.829 | 0.727 | 0.677 | 0.694 | 0.699 | 0.672 |
| CHE | 1.49 | 0.85 | 1.13 | 1.39 | 1.48 | 1.47 | 1.32 |
| CC | 6.04 | 3.25 | 3.55 | 5.23 | 5.65 | 5.67 | 4.81 |
| CTnCTR | 0.911 | 0.929 | 0.883 | 0.880 | 0.875 | 0.877 | 0.888 |
| PCS | 0.505 | 0.628 | 0.530 | 0.482 | 0.472 | 0.470 | 0.506 |
| MCTD | 1.338 | 1.262 | 1.353 | 1.372 | 1.378 | 1.378 | 1.355 |
| 조성 일관성 | 0.855 | 0.922 | 0.948 | 0.894 | 0.830 | 0.829 | 0.914 |
| 반복 프레이즈 일치 | 0.856 | 0.977 | 0.984 | 0.879 | 0.871 | 0.869 | 0.896 |
| π_ref^(1/0.85)와의 곡당 KL | – | – | – | 0 | 0.257 | 0.244 | 0.215 |

RL 역방향·순방향은 최종 보상으로 β 0.2, RL 국소는 국소 보상으로 β 0.8입니다. VALID 결과도 같은
양상이어서, VALID에서 한 선택이 VALID에 과적합하지 않았습니다.

- 최종 보상은 요구한 대로 움직였습니다. 조성 밖 시간이 T 0.85 π_ref의 두 배로 사람보다 많고, 긴
  지속은 가장 적으며, 코드 다양성(CHE, CC)은 사람에 가깝습니다.
- 국소 보상은 반대로 움직였습니다. 더 협화적이고(PCS), 조성 밖 코드가 적고, 덜 다양하며, 지속이
  깁니다.
- 두 KL 방향은 모든 지표에서 거의 같습니다. β 0.2와 날카롭게 한 참조에서 둘 다 곡당 KL 약 0.25에
  머물렀습니다.

### 청취

| 회차 | 곡 | 시스템 | 결과 |
|---|---|---|---|
| 5 | 20 | T 0.85, 역방향, 순방향 | 수용 가능 11 / 14 / 14. 역방향이 T 0.85보다 위 20곡 중 13곡, 순방향이 역방향보다 위 19곡 중 8곡. 모두 유의하지 않음. |
| 6 | 20 | T 0.85, 역방향, 국소 | 수용 가능 15 / 10 / 16. 국소가 역방향보다 위 20곡 중 12곡(H1). 5·6회차 합: 역방향이 T 0.85보다 위 40곡 중 21곡. |
| 7, 1단계 | 10 × 뽑기 3번 | T 0.85, 순방향, 역방향, 국소 | 수용 가능한 뽑기 30번 중 20 / 16 / 16 / 19. 문제로 표시된 마디 5.7 / 18.8 / 19.5 / 6.9%. |
| 7, 2단계 | 10 | 시스템별 최고 뽑기 | 평균 순위 2.6 / 2.6 / 2.4 / 2.4. 모든 쌍이 10곡 중 4–6곡. |

다시 뽑아서 3개 중 최고를 고를 수 있으면 네 시스템은 구분되지 않았고, 역방향 모델의 최고 뽑기는
T 0.85의 두 배인 조성 밖 시간을 유지했습니다(10.4% 대 6.0%). 뽑기 하나 단위로는 덜 믿을 만합니다.
대부분의 회차에서 조성 밖 보상이 조성 밖으로 더한 마디의 약 40%가 문제로 표시됐고(7회차 41–47%,
5회차 7%), 문제의 상당수는 1마디짜리였습니다. RL은 정책이 조성 밖으로 더 자주 나가게 했을 뿐 어디서
나갈지를 더 잘 고르게 하지는 않았습니다. π_ref는 장르에 따라 어디가 안전한지 알 수 없고, 장르는
입력에 없습니다.

## 배포 모델

| 모델 | 출처 실행 | 기본 온도 | 스타일 |
|---|---|---|---|
| [`harmonyrl-chromatic`](https://huggingface.co/CloudHolic/harmonyrl-chromatic) | RL 역방향, β 0.2, 최종 보상 | 1.0 | 조성 밖 코드가 많은 풍부한 리하모니제이션 |
| [`harmonyrl-smooth`](https://huggingface.co/CloudHolic/harmonyrl-smooth) | RL 국소, β 0.8 | 1.0 | 협화적이고 해결되는 진행 |
| [`harmonyrl-base`](https://huggingface.co/CloudHolic/harmonyrl-base) | π_ref, 사전학습만 | 0.85 | 학습 데이터에 가까움 |

순방향 KL 실행은 H2 결과로만 남겼습니다. 세 모델 모두 다시 뽑기를 전제로 씁니다. 라이브러리는 서로
다른 후보 여러 개를 돌려주고, 고른 마디만 다시 뽑습니다.

## 재현

### 환경

- Python 3.14, [uv](https://docs.astral.sh/uv/)로 관리: `uv sync`로 전부 설치하고 `uv run`으로
  스크립트를 실행합니다. 노트북에 필요한 패키지는 `notebook` 의존성 그룹에 있습니다.
- 개발 환경: NVIDIA GB10(aarch64, CUDA 13.0, driver 580). PyTorch는 PyPI에서 받습니다.
- aarch64에서는 `uv run`이 호출마다 `nvidia-cusparselt-cu13`을 다시 설치할 수 있고, 이때 이미
  실행 중인 다른 `uv run`이 깨집니다. 오래 걸리는 작업은 `uv run --no-sync`로 시작합니다.
- 렌더링과 청취 도구에는 [FluidSynth](https://www.fluidsynth.org)와 `FluidR3_GM.sf2`, ffmpeg,
  `mscore`로 실행되는 [MuseScore 4](https://musescore.org)가 필요합니다. 경로는
  `configs/rendering.yaml`과 `configs/build_round.yaml`에 있습니다. 학습과 평가에는 필요 없습니다.

### 데이터

```sh
git clone https://github.com/chrisdonahue/sheetsage-data
git -C sheetsage-data checkout 06113c0
mkdir -p data/raw/hooktheory
cp sheetsage-data/hooktheory/Hooktheory.json.gz sheetsage-data/hooktheory/Hooktheory_Raw.json.gz data/raw/hooktheory/
```

SHA-256:

```
917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c  Hooktheory.json.gz
716af2979f060400c302ab098dd45d9f8c5fe4d4b3b1fe61c478dd9bdf041634  Hooktheory_Raw.json.gz
```

### 실행

모든 스크립트는 [Hydra](https://hydra.cc)로 설정합니다. 기본값은 `configs/<스크립트>.yaml`에 있고,
덮어쓰기는 `key=value`, 합성된 설정 출력은 `--cfg job`, sweep은 `-m`입니다. 실행마다 `output_dir`
(기본은 `runs/` 아래, 이름은 `name=` 또는 시작 시각)에 쓰고, 비어 있지 않은 디렉터리는 거부하며,
합성된 설정과 git 해시, 지표를 저장합니다.

```sh
# 세 분할의 구간을 data/processed에
uv run scripts/data/prepare_data.py

# π_ref. 최고 epoch는 runs/pretrain/final/best에 저장
uv run --no-sync scripts/train/pretrain.py name=final

# HMM 기준선
uv run scripts/train/train_hmm.py name=base

# RL
uv run --no-sync scripts/train/train_rl.py name=reverse-b0.2
uv run --no-sync scripts/train/train_rl.py name=forward-b0.2 train.divergence=forward
uv run --no-sync scripts/train/train_rl.py name=local-b0.8 train.reward=local train.beta=0.8
```

평가는 실행 하나에 시스템 하나를 채점합니다. `rewards`는 함께 채점할 선호 보상 모델 목록이며,
없으면 지웁니다.

```sh
uv run --no-sync scripts/eval/evaluate.py split=test name=human system=human '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=hmm system=hmm '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=pi_ref_greedy system=policy system.greedy=true '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=pi_ref_t0.85 system=policy '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=rl_reverse_b0.2 system=policy \
	system.model=runs/rl/reverse-b0.2/final system.temperature=1.0 '~rewards.reward'
```

순방향과 국소 실행도 같은 방식으로 평가합니다. `notebooks/results.ipynb`가 평가 디렉터리를 읽어
표와 그림을 그립니다.

학습한 정책을 기본 온도와 함께 배포 모델로 만드는 방법:

```sh
uv run scripts/train/release.py --source runs/rl/reverse-b0.2/final --output runs/release/chromatic
uv run scripts/train/release.py --source runs/pretrain/final/best --output runs/release/base --temperature 0.85
```

선호 보상(`scripts/train/train_reward.py`)과 청취 도구에는 소유자의 판정이 필요하며, 판정은
공개하지 않습니다.

### 코드 검사

```sh
uv run ruff check
uv run ruff format --check
uv run ty check
```

## 저장소 구조

```
configs/                  스크립트별 Hydra 설정. system/은 평가할 시스템
scripts/
  data/                   데이터셋 준비와 통계, 청취 회차 만들기와 수집
  train/                  사전학습, HMM, 선호 보상, RL, 배포
  eval/                   분할 하나에서 시스템 하나 평가
  io/                     렌더링, 가져온 파일 확인
  tools/                  청취 도구(쌍·순위 판정, 둘러보기)
src/harmonyrl_research/
  schema/                 데이터 계약: 멜로디, 코드 어휘, 박자, 조성, 행동 마스크
  data/                   원본 파일 결합, 필터링, 변환
  metrics/                Yeh et al., 구조, 어법, 다양성, 보상 항
  models/                 공통 특징, 배치, 어텐션. policy/, hmm/, reward/
  rl/                     rollout, 보상, RLOO·DPG 학습기
  ingest/                 MusicXML, MIDI, MuseScore, PDF, 단선율 음원을 멜로디로
  render/                 MIDI, WAV, MP3, MusicXML, PDF, MuseScore 출력
  tools/                  청취 도구
  harmonizer.py           배포 모델로 멜로디 전체 또는 고른 마디만 하모나이제이션
notebooks/results.ipynb   결과 표와 그림
```

`data/`와 `runs/`는 추적하지 않습니다.

## 한계

- **청취자 1인.** 선택은 한 사람의 귀를 반영합니다. 그 청취자가 무엇을 좋아했는지를 보여줄 뿐,
  일반적인 선호가 아닙니다.
- **장르가 입력에 없습니다.** 조성 밖 코드가 어울리는지는 장르에 달려 있지만 모델은 알 수
  없습니다.
- **얼마나 자주가 아니라 어디서.** RL은 정책이 조성을 벗어나는 빈도를 높였을 뿐, 어디서 벗어날지를
  더 잘 고르게 하지는 않았습니다. RL 모델의 뽑기 하나는 π_ref보다 다시 뽑아야 할 때가 많습니다.
- **32마디 문맥.** 한 번에 32마디까지 다루고, 더 긴 곡은 겹치는 창으로 나눠 처리하므로 반복되는
  절 같은 곡 전체의 구조는 모델링하지 않습니다.
- **학습 데이터**는 모든 장르의 곡을 사용자가 받아 적은 것이며, 좋은 반주만 골라 모은 것이
  아닙니다.

## 라이선스와 참고 문헌

코드는 MIT 라이선스입니다. 모델 가중치와 데이터셋에서 나온 모든 것은 Hooktheory 데이터를 따라
CC BY-NC-SA 3.0입니다. 데이터는 이 저장소에서 추적하지 않습니다.

- Donahue, Thickstun, Liang. "Melody Transcription via Generative Pre-Training." ISMIR 2022.
  [arXiv:2212.01884](https://arxiv.org/abs/2212.01884). 데이터 공개본의 출처.
- Yeh et al. "Automatic Melody Harmonization with Triad Chords: A Comparative Study."
  [arXiv:2001.02360](https://arxiv.org/abs/2001.02360). 지표와 HMM 기준선.
- Ji, Yang, Luo, Li. "RL-Chord: CLSTM-Based Melody Harmonization Using Deep Reinforcement
  Learning." IEEE Transactions on Neural Networks and Learning Systems, 2024.
  [doi:10.1109/TNNLS.2023.3248793](https://doi.org/10.1109/TNNLS.2023.3248793). 국소 보상.
- Ahmadian et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
  Human Feedback in LLMs." [arXiv:2402.14740](https://arxiv.org/abs/2402.14740). RLOO.
- Parshakova, Andreoli, Dymetman. "Distributional Reinforcement Learning for Energy-Based
  Sequential Models." [arXiv:1912.08517](https://arxiv.org/abs/1912.08517). DPG.
- Khalifa, Elsahar, Dymetman. "A Distributional Approach to Controlled Text Generation."
  ICLR 2021. [arXiv:2012.11635](https://arxiv.org/abs/2012.11635).
- Korbak, Elsahar, Kruszewski, Dymetman. "Controlling Conditional Language Models without
  Catastrophic Forgetting." [arXiv:2112.00791](https://arxiv.org/abs/2112.00791). 조건부 DPG.
- Go et al. "Aligning Language Models with Preferences through f-divergence Minimization."
  [arXiv:2302.08215](https://arxiv.org/abs/2302.08215). 같은 목표를 향한 KL 방향 비교.
- Zeng et al. "MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training."
  [arXiv:2106.05630](https://arxiv.org/abs/2106.05630). 폐기한 보상의 특징.
