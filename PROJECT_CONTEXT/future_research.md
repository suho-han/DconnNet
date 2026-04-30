# Future Research

연구적으로 진행해야 하는 실험 및 구조 개선 사항을 시도 순서대로 나열함.

## dg / dg_direct

- 기존 `decoder_guided`는 canonical 이름을 `dg`로 정리
- 기존 산출물 폴더도 `_decoder_guided_` -> `_dg_` rename
- 새 변형 `dg_direct`는 fused connectivity는 `dg`와 동일하게 두고, `SegAux` mask head만 `final_feat` 단독 입력으로 비교

## segaux의 w(`seg_aux_weight`) 범위

1, 5, 10 정도 시도해보고 적정값 찾기

## inner8, outer8

현재 inner8(3x3), outer8(5x5)의 상하좌우+대각선만 활용하는 것에 추가하여 5x5 내의 남은 8개의 픽셀도 그룹을 만들어서 활용할 계획

## 5x5 -> 7x7 더 크게 확장
