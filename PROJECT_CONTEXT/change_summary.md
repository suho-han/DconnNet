# `c33561d`부터 현재까지 변경 요약

Last updated: 2026-04-29

## 2026-04-30 naming update

- legacy `decoder_guided`는 canonical 이름 `dg`로 정리합니다.
- 새 `dg_direct`는 fused connectivity는 `dg`와 같고, `SegAux` mask head만 `final_feat` 단독 입력을 사용합니다.
- 기존 `*_decoder_guided_*` 결과 폴더는 `*_dg_*`로 rename 대상입니다.

## 범위

- 기준 커밋: `c33561ded94368a51a741bbb105b40d533051a2a`
- 기준 커밋 메시지: `feat(core): add 24to8 grouping and dist-aux modular losses`
- 이 문서는 2026-04-28 기준 현재 working tree까지 포함한 요약입니다.
- 정리 대상은 `24to8 grouping`, `encoder fusion`, `segaux` 세 가지입니다.

## 읽는 방법

- 이 문서에서 `encoder fusion`은 코드상의 `conn_fusion` 경로를 뜻합니다.
- 이름과 달리 실제 결합 지점은 backbone encoder 내부가 아니라 `final_decoder` 이후의 directional logits 단계입니다.
- 현재도 `24-direction connectivity` 자체는 남아 있습니다. 제거된 것은 `24to8 grouping`이라는 별도 reducer 경로입니다.

## 큰 흐름

- `c33561d` 시점에는 `24to8 grouping`이 별도 기능으로 존재했습니다.
- 이후 이 grouped reducer는 제거되었습니다.
- 대신 현재 포크는 다음처럼 기능을 분리해서 확장합니다.
  - 어떤 방향 집합을 예측할지: connectivity layout
  - 둘 이상의 directional prediction을 어떻게 결합할지: `conn_fusion`
  - segmentation 보조 supervision을 줄지: `SegAux`

## 1. `24to8 grouping`

### 과거 역할

- `24to8 grouping`은 더 촘촘한 방향 정보를 먼저 만들고, 그것을 canonical 8-direction 표현으로 줄이는 별도 reducer 경로였습니다.
- 즉, 방향 수를 늘리는 것과 최종 출력 채널 수를 유지하는 것을 동시에 만족시키기 위한 중간 표현 계층이 있었습니다.

### 제거된 이유와 현재 의미

- 현재 포크는 이 중간 reducer를 유지하지 않습니다.
- 방향 집합을 줄이기 위한 별도 그룹화 대신, 처음부터 어떤 방향 집합을 쓸지 명시적으로 선택하는 구조로 단순화되었습니다.
- 결과적으로 모델은 다음 둘을 분리해서 다룹니다.
  - `standard8`: 기존 8방향 connectivity
  - `out8`: 바깥쪽 8방향 connectivity
- 따라서 지금은 `24 -> 8`로 축약하는 개념보다, 서로 다른 8방향 표현을 나란히 만들고 필요하면 fusion하는 개념이 중심입니다.

### 현재 구조에서 남은 것과 사라진 것

- 남은 것:
  - `8-direction` 경로
  - `24-direction` 경로 자체
  - outer-neighborhood를 명시적으로 쓰는 `out8` 개념
- 사라진 것:
  - `24to8` 전용 reducer
  - direction-grouping 전용 실행 흐름
  - grouped path를 전제로 한 전용 메타데이터 처리

### 개념적으로 대체된 방식

- 예전에는 `24채널을 만든 뒤 8채널로 축약`했다면,
- 지금은 `inner 8방향`과 `outer 8방향`을 별도 head로 만들고,
- 두 표현을 직접 결합해 최종 connectivity를 얻습니다.
- 즉, 축약 중심 설계에서 병렬 표현 + 결합 설계로 바뀌었습니다.

## 2. `encoder fusion` (`conn_fusion`)

### 핵심 아이디어

- `conn_fusion`은 서로 다른 방향성 정보를 하나의 connectivity prediction으로 합치는 장치입니다.
- 현재 구현에서 이 두 정보원은 다음과 같습니다.
  - `C3`: inner/standard 방향성 logits
  - `C5`: outer 방향성 logits
- 중요한 점은 단순히 채널을 이어붙이는 것이 아니라, 두 표현이 같은 방향 의미를 가지도록 먼저 정렬한 뒤 결합한다는 것입니다.

### 내부 데이터 흐름

1. `final_decoder`의 출력을 바탕으로 두 개의 directional head를 만듭니다.
2. 첫 번째 head는 `standard8` 의미를 갖는 `C3`를 만듭니다.
3. 두 번째 head는 `out8` 의미를 갖는 `C5`를 만듭니다.
4. 두 branch를 동일한 방향 의미 위에서 해석할 수 있도록 맞춘 뒤 chosen fusion rule로 결합해 `C_fused`를 만듭니다.

### 옵션별 구현 개념

- `none`
  - 별도 fusion을 하지 않습니다.
  - 기존 single connectivity head 경로를 그대로 사용합니다.
  - 한 줄 수식: `C_fused = C_base`

- `gate`
  - 핵심 아이디어: `C3`와 `C5_align`를 함께 보고 위치/채널별 gate를 예측해 두 branch를 convex combination으로 섞습니다.
  - 한 줄 수식: `alpha = σ(g([C3, C5_align])),  C_fused = alpha ⊙ C3 + (1-alpha) ⊙ C5_align`
  - 코드 기준 정의: `g = Conv1x1`, 즉 `fusion_gate_conv = nn.Conv2d(out_planes*2, out_planes, 1)`입니다.
  - shape/의미: `alpha`는 `C3`와 같은 shape `(B, out_planes, H, W)`이며 방향/위치별 혼합 비율입니다.

- `scaled_sum`
  - 핵심 아이디어: `C3`를 주 경로로 두고 `C5_align`를 고정 스케일 residual처럼 더합니다.
  - 한 줄 수식: `C_fused = C3 + λ · C5_align`
  - 코드 기준 정의: `λ = fusion_residual_scale` (기본값 `0.2`)로, 학습되는 gate 없이 상수 계수로 결합합니다.
  - shape/의미: `C3`, `C5_align`, `C_fused`는 동일 shape이며 outer branch는 보정 신호로만 작동합니다.

- `conv_residual`
  - 핵심 아이디어: `C5_align`를 Conv로 변환한 residual을 `C3`에 더해, outer 정보를 `C3`에 맞는 보정량으로 사용합니다.
  - 한 줄 수식: `R = Conv(C5_align),  C_fused = C3 + R`
  - 코드 기준 정의: `Conv = fusion_residual_conv = nn.Conv2d(out_planes, out_planes, 1)`입니다.
  - shape/의미: `R`은 `C3`와 같은 shape로 변환되어 채널/위치별 additive correction 역할을 합니다.

- `decoder_guided`
  - 핵심 아이디어: `C3`, `C5_align`뿐 아니라 decoder 문맥(`F_dec`)까지 함께 사용해 residual과 gate를 동시에 생성합니다.
  - 한 줄 수식: `(R, β) = h(C3, C5_align, F_dec),  C_fused = C3 + β ⊙ R`
  - 코드 기준 정의: `F_dec`는 `final_feat` 자체가 아니라 `final_feat_up = upsample2x(final_feat)`이며, `x = cat([F_dec, C3, C5_align])`, `h = ReLU(BN(Conv3x3(x)))`, `β = Sigmoid(Conv1x1(h))`, `R = Conv1x1(h)`입니다.
  - shape/의미: `F_dec(=final_feat_up)`는 32채널 decoder feature, `C3/C5_align`는 `out_planes` 채널 connectivity logits이며, 같은 해상도에서 concat된 `h(hidden=64)`로부터 `β`(보정 강도)와 `R`(보정 내용)을 생성합니다.
  - 텐서 차원:
    - `F_dec`: `(B, 32, H, W)`
    - `C3`: `(B, out_planes, H, W)`
    - `C5_align`: `(B, out_planes, H, W)`
    - `x = cat([F_dec, C3, C5_align], dim=1)`: `(B, 32 + 2*out_planes, H, W)`
    - `h`: `(B, 64, H, W)`
    - `β`: `(B, out_planes, H, W)`
    - `R`: `(B, out_planes, H, W)`
    - `C_fused`: `(B, out_planes, H, W)`
  - 현재 fusion 제약(`num_class=1`, `conn_num=8`)에서는 `out_planes=8`이므로 `x=(B,48,H,W)`, `h=(B,64,H,W)`, `β/R/C_fused=(B,8,H,W)`입니다.

### 학습 신호가 들어가는 방식

- `conn_fusion`이 켜지면 단순히 최종 fused branch만 학습시키지 않습니다.
- 기본적으로는 fused connectivity에 대해 segmentation 관련 항과 affinity 관련 항이 주어집니다.
- 그리고 profile에 따라 inner/outer branch의 affinity 항을 추가할 수 있습니다.
- 이 구조는 fused 결과를 메인으로 두되, 각 branch가 자기 의미를 잃지 않도록 약한 보조 제약을 거는 방식입니다.

### profile A/B/C의 의미

- `A`
  - fused 결과만 중심으로 학습합니다.
  - 가장 공격적으로 fused output 하나에 수렴시키는 설정입니다.
  - 한 줄 수식: `L_A = L_base(C_fused) + λ_fused·L_aff(C_fused)`
  - 코드 대응: `L_base(C_fused) = L_vote(C_fused)+L_dice(C_fused)`

- `B`
  - `A`에 inner branch affinity 제약을 추가합니다.
  - 즉, fused 결과를 학습하면서도 inner 표현이 connectivity 의미를 유지하도록 붙잡아 둡니다.
  - 한 줄 수식: `L_B = L_A + λ_inner·L_aff(C3)`

- `C`
  - `B`에 outer branch affinity 제약까지 더합니다.
  - 세 branch가 모두 구조적 의미를 유지하도록 가장 강하게 제약하는 설정입니다.
  - 한 줄 수식: `L_C = L_B + λ_outer·L_aff(C5_native)`

### `L_vote` vs `L_dice` 차이
- 역할 차이
  - `L_vote`는 bilateral voting으로 얻은 mask score(`pred`)를 픽셀 단위로 맞추는 항입니다.
  - `L_dice`는 같은 `pred`를 겹침 비율(집합 유사도) 관점에서 맞추는 항입니다.
- 한 줄 수식(개념)
  - `L_vote = BCE(pred, y)`  (dist 모드에서는 `dist_aux_regression_loss(pred, y)` 사용)
  - `L_dice = 1 - Dice(pred, y) = 1 - (2|pred∩y|+ε)/(|pred|+|y|+ε)`
- 코드 대응
  - `binary` 라벨 모드: `vote -> bce_loss`, `dice -> dice_l` (`connect_loss.single_class_forward`)
  - `dist/dist_inverted` 모드: `vote -> vote_loss(dist_aux_regression_loss)`, `dice -> dice_l`

### `dist/dist_inverted` 손실 조합 (코드 기준)

- 타깃 해석
  - `mask_target = (target > 0).float()`로 vessel mask를 복원합니다.
  - 거리맵 `target`은 최종 binary path supervision에 직접 쓰지 않고, affinity regression target 생성 쪽 의미를 유지합니다.
- 기본 항
  - `L_vote = mean(dist_aux_regression_loss(pred, mask_target))`
  - `L_dice = DiceLoss(pred[:,0], mask_target[:,0])`
  - `L_edge = dist_edge_loss(bicon_map, mask_target)`
  - `L_aff = dist_aux_regression_loss(c_map, affinity_target)`
- `dist_aux_loss='cl_dice'` 예외
  - dense affinity map `(B,8,H,W)`에 직접 clDice를 걸면 메모리 사용량이 커지므로, `L_vote` 쪽만 clDice 계열 supervision을 유지합니다.
  - 이 경우 affinity/bicon regression은 둘 다 `SmoothL1`로 강제됩니다.
  - 한 줄 수식:
    - `L_aff = SmoothL1(c_map, affinity_target)`
    - `L_bicon = SmoothL1(bicon_map, affinity_target)` (`CHASE` 제외)
- 데이터셋별 최종 합산식
  - `CHASE`:
    - `L_bicon = 0`
    - `L_total = L_vote + L_aff + L_edge + L_dice`
  - 그 외 데이터셋:
    - `L_bicon = dist_aux_regression_loss(bicon_map, affinity_target)` 또는 `SmoothL1(...)` (`cl_dice` 예외)
    - `L_total = L_vote + L_aff + L_edge + 0.2·L_bicon + L_dice`
- 구현상 의미
  - `pred`는 distance mode에서도 여전히 최종 vessel mask를 예측하는 binary path입니다.
  - `c_map`과 `bicon_map`은 mask 자체가 아니라 distance-derived affinity target에 맞춰 회귀됩니다.
  - edge 통계 수집(`_collect_dist_edge_stats`)은 로깅 전용이며, 손실식 자체는 바꾸지 않습니다.

### Bilateral Voting 계산 흐름 (코드 기준)
- 입력 텐서
  - `c_map`: directional connectivity 확률맵 (`sigmoid` 이후), shape `(B, conn_num, H, W)`
  - 내부에서 `class_pred = c_map.view(B, num_class, conn_num, H, W)`로 변환
- 핵심 연산
  - 각 방향 채널을 반대 방향으로 shift/translate한 값과 곱해 방향별 합치성 `vote_out`을 만듭니다.
  - 한 줄 수식: `vote_out[k] = c_k ⊙ shift(c_rev(k))`
  - 최종 score map은 채널축 max 집계: `pred = max_k vote_out[k]`
- 출력
  - `pred`: `(B, num_class, H, W)` (vote 기반 segmentation score map)
  - `vote_out`(또는 `bicon_map`): `(B, num_class, conn_num, H, W)` (방향별 voting 결과)
- 손실 연결
  - `L_vote`와 `L_dice`는 모두 이 `pred`에 걸립니다.
  - `L_aff`는 `c_map`과 affinity target(`con_target`/`affinity_target`) 사이에서 계산됩니다.
- 해상도 규칙
  - voting은 `c_map`의 현재 해상도 `H×W`에서 수행됩니다.
  - 따라서 손실 계산 시 `target`도 같은 `H×W`여야 하며, mismatch 시 shape 오류가 납니다.

### 해상도 변경은 어디서 일어나는가
- `sigmoid`는 값 범위만 바꾸고 shape는 유지합니다. 해상도 변경은 모델(`model/DconnNet.py`)의 upsample/interpolate에서 처리됩니다.
- `C3/C5` 업샘플: `c3_logits = upsample2x(c3_logits)`, `c5_logits_native = upsample2x(c5_logits_native)` 후 voting/loss로 전달
- `decoder_guided`용 `F_dec`: `final_feat_up = upsample2x(final_feat)`를 `decoder_guided_fusion`에 입력

### `decoder_guided`에서 추가되는 요소

- 이 모드는 단순 fusion보다 구조가 복잡하므로 solver 쪽에서도 추가 보조항을 붙입니다.
- `C3`와 `C5` branch 자체에 auxiliary loss를 줄 수 있고,
- gate 맵의 평균값에 대한 regularization도 줄 수 있습니다.
- 즉, fused 결과만 좋게 만드는 것이 아니라, residual을 만드는 두 재료와 gate의 동작까지 함께 안정화하려는 설계입니다.

### 출력 계약의 변화

- `conn_fusion=none`일 때는 기존과 같은 단순 출력 구조를 유지합니다.
- `conn_fusion`이 활성화되면 모델은 최소한 다음 의미를 가진 출력을 제공합니다.
  - fused connectivity
  - inner branch connectivity
  - outer branch connectivity
  - 정렬된 outer connectivity
  - auxiliary feature map
- 즉, 현재 fusion 경로는 단일 출력 모델이 아니라 다중 branch를 노출하는 분석 가능한 구조입니다.

## 3. `segaux` (`use_seg_aux`)

### 핵심 아이디어

- `SegAux`는 connectivity 기반 표현과 decoder feature를 함께 보고, 별도의 segmentation 보조 head에서 추가 supervision을 주는 장치입니다.
- 목표는 메인 경로를 대체하는 것이 아니라, connectivity가 실제 mask 구조와 더 직접적으로 연결되도록 보조하는 것입니다.
- 한 줄 수식: `L_total = L_main + λ_aux · BCE(mask_logit, Y_mask)`
- 코드 기준 정의: `mask_logit = ConnectivitySegHead([Up(final_feat), C_fused])`, `ConnectivitySegHead = Conv3x3(dec_ch+conn_ch→hidden)+BN+ReLU+Conv1x1(hidden→1)`입니다.
- shape/의미: `mask_logit`는 `(B,1,H,W)`이며, fused connectivity가 mask space와 직접 정렬되도록 보조 감독합니다.

### 내부 데이터 흐름

1. decoder feature를 upsample합니다.
2. 여기에 fused connectivity를 함께 붙입니다.
3. `ConnectivitySegHead`가 이를 받아 `mask_logit`을 만듭니다.
4. 이 `mask_logit`에 대해 BCE 기반 보조 손실을 계산합니다.

### 왜 필요한가

- connectivity loss만으로 학습하면 모델은 연결 구조는 맞추더라도 실제 mask 경계나 채움 패턴과 느슨하게 연결될 수 있습니다.
- `SegAux`는 connectivity 표현이 실제 segmentation target과 직접 대응되도록 한 번 더 끌어당깁니다.
- 즉, 구조 표현과 mask 표현 사이의 거리를 줄이는 bridge supervision 역할입니다.

### 메인 경로와의 관계

- `SegAux`는 standalone segmentation branch가 아닙니다.
- 메인 connectivity 학습 위에 얹히는 auxiliary term입니다.
- 그래서 이 기능이 켜져도 fused connectivity가 주 경로라는 사실은 유지됩니다.
- 다만 segmentation 형태에 대한 직접 감독이 추가되므로, connectivity 표현이 더 mask-friendly한 방향으로 정리될 가능성이 있습니다.
- 최종 loss 수식(`SegAux` 포함):
  - `L_total = L_main + λ_segaux·BCE(mask_logit, Y_mask)`
  - 여기서 `L_main = (L_vote+L_dice) + λ_fused·L_aff + profile(B/C 항)`입니다.

### legacy path에서의 의미

- fusion이 없는 경로에서도 `SegAux`는 사용할 수 있습니다.
- 이 경우에는 기존 connectivity output 자체를 segmentation head와 함께 해석해 보조 손실을 줍니다.
- 즉, `SegAux`는 fusion 전용 기능이 아니라, connectivity representation 전반에 붙일 수 있는 segmentation regularizer입니다.

### `decoder_guided`과의 관계

- `decoder_guided`은 decoder context를 이용해 fused connectivity를 만드는 기능입니다.
- `SegAux`는 그렇게 만들어진 fused connectivity가 실제 mask와 잘 맞도록 추가 감독을 줍니다.
- 따라서 둘은 경쟁 관계가 아니라 `어떻게 fused connectivity를 만들 것인가`와 `그 fused connectivity가 segmentation과 얼마나 직접 맞닿아야 하는가`를 나눠 다루는 서로 다른 층위의 옵션입니다.

## 4. 세 옵션을 함께 보면 구조가 어떻게 바뀌는가

- `24to8 grouping` 제거 이후 핵심 변화는 `중간 축약기`가 사라졌다는 점입니다.
- 현재 구조는 `layout 선택 -> conn_fusion -> SegAux`의 3단계로 이해하는 편이 정확합니다.
- 즉, 현재 포크의 설계는 `방향 수 축약`보다 `표현 분리 -> 표현 결합 -> 보조 감독`의 단계적 구조에 가깝습니다.

## 5. 현재 스냅샷 검증 상태

- 2026-04-28 기준 관련 회귀 테스트는 `47 passed`였고, warning은 있었지만 정상 종료되었습니다.

## 5-1. 문서-코드 정합성 재검증 (2026-04-29)

- 확인 범위: `model/DconnNet.py`, `solver.py`, `connect_loss.py`
- 검증 결과(반영 완료):
  - `F_dec`는 `final_feat`가 아니라 `final_feat_up = upsample2x(final_feat)`로 사용됨 (`model/DconnNet.py` 339-340)
  - outer branch loss는 `outer_aligned`가 아니라 `outer`(= `c5_logits_native`) 기준으로 계산됨 (`model/DconnNet.py` 354, `solver.py` 645)
  - profile 손실 조합은 `compose_fusion_profile_loss_terms` 식과 일치함 (`solver.py` 51-129)
  - `decoder_guided` 추가 보조항은 `conn_aux_c3_weight * inner_terms['total|affinity']` + `conn_aux_c5_weight * outer_terms['total|affinity']` + gate regularization 구조임 (`solver.py` 671-684)

## 6. 구현을 볼 때 핵심이 되는 파일

- `model/DconnNet.py`: directional head 생성, `conn_fusion`/`SegAux` head 구현
- `solver.py`: fusion profile 손실 조합, `decoder_guided` 보조항, `SegAux` 보조 BCE 결합
- `connect_loss.py`: connectivity layout과 directional 의미 정의

## 7. 모델 아키텍처 요약

### 기존 DconnNet 구조는 간단히 보면

```text
Input x
  |
  v
ResNet34 backbone
  |
  +-> c1, c2, c3, c4, c5
  |
  +-> c5 -> directional prior -> SDE
  |
  +-> decoder (SpaceBlock + FeatureBlock + final_decoder)
  |
  `-> final_feat -> single connectivity head -> fused
```

### 바뀐 부분만 강조한 구조

```text
final_feat
  |
  +-- [기존] single head -----------------------------------------------> fused
  |
  +-- [변경 1] conn_fusion
  |      |
  |      +-> inner head ---------------------------------------------> C3
  |      +-> outer head ---------------------------------------------> C5
  |      `-> fusion(C3, C5[, decoder context]) ---------------------> C_fused
  |
  `-- [변경 2] SegAux
         |
         +-> upsample(final_feat)
         `-> ConnectivitySegHead(upsampled final_feat, connectivity) -> mask_logit
```

### 병렬 여부를 정확히 보면

- `final_feat`에서 바로 갈라지는 `inner head`와 `outer head`는 병렬입니다.
- 하지만 두 head는 독립 최종 출력으로 끝나지 않고, 하나의 `fused/C_fused`로 합쳐집니다.
- `SegAux`는 이 `fused/C_fused`를 다시 입력으로 사용하므로, `inner/outer`와 완전히 독립인 병렬 head는 아닙니다.
- `aux = mapped_c5`는 `final_decoder` 이후 branch가 아니라, 더 앞쪽의 `c5 -> channel_mapping -> upsample` 경로에서 나오는 별도 auxiliary branch입니다.

### 바뀐 부분만 강조한 loss 구조

```text
[기존]
L_total = L_conn(fused) + 0.3 * L_conn(mapped_c5)

[변경 1] conn_fusion
fused main = (vote + dice) + lambda_fused * affinity(fused)

Profile A:
  L_total = fused main

Profile B:
  L_total = fused main + lambda_inner * affinity(inner)

Profile C:
  L_total = fused main
          + lambda_inner * affinity(inner)
          + lambda_outer * affinity(outer)

[변경 2] SegAux
위 어떤 경로 위에도 추가 가능:
  L_total = L_total + seg_aux_weight * BCE(mask_logit, binary_gt)

[변경 1의 특수형] decoder_guided
  L_total = L_total
          + conn_aux_c3_weight * L_conn(C3)
          + conn_aux_c5_weight * L_conn(C5_native)
          + gate regularization
```

### 중간 feature와 출력 의미

- `final_feat`: 기존 DconnNet의 최종 공용 decoder 표현입니다. 포크의 모든 변경점은 이 feature 이후에 붙습니다.
- `C3`: 새로 추가된 inner directional branch 출력입니다.
- `C5`: 새로 추가된 outer directional branch 출력입니다.
- `C_fused`: 포크에서 새로 정의한 최종 connectivity 결합 결과입니다.
- `aux(mapped_c5)`: `final_feat` 이후가 아니라 깊은 `c5` branch에서 오는 기존 auxiliary 표현입니다.
- `mask_logit`: `SegAux`가 켜졌을 때만 생기는 보조 segmentation 출력입니다.

## 2026-04-29 Update: CHASE scaled_sum residual ablation pending-only config

- `scripts/configs/drive_chase_scaled_sum_residual_ablation.yaml`를 미완료 조합만 남기도록 정리했다.
- 완료된 `drive` 조합과 완료된 `chase` 조합을 제외하고, 아래 9개만 남긴 상태다.
  - `chase`: `dist + gjml_sf_l1 + rs0.5` (1개)
  - `chase`: `dist_inverted + smooth_l1 + rs{0.1,0.2,0.3,0.5}` (4개)
  - `chase`: `dist_inverted + gjml_sf_l1 + rs{0.1,0.2,0.3,0.5}` (4개)
