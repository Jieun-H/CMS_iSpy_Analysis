# ML 작업 인계 문서

PointCloud_Agent로 만든 CMS 점구름(npz)을 머신러닝에 쓰기 위한 정리입니다.
데이터 생성 쪽 작업은 끝났고, 이 문서는 모델 학습을 시작하는 다음 작업을 위한 것입니다.
기준 커밋: `5ee21d8` (2026-10-14).

---

## 1. 연구 목표와 첫 과제

- **최종 목표**: 통계분석이 잡지 못하는 패턴을 검출기 데이터에서 직접 찾는 이상탐지.
- **첫 과제 (가장 단순한 검증)**: 이벤트에 뮤온이 있는지를 모델이 검출기 신호만으로 알아내는가.
  - 뮤온은 다른 트랙보다 멀리 뻗어 뮤온 챔버(CSC/DT/RPC)에 흔적을 남긴다는 점이 핵심 단서.
- **평가 계획**: DY→μμ로 학습한 뒤, DY→ee / eμ / μμ 이벤트에서 뮤온이 있는 이벤트를 찾아내는지 확인.
  - 이벤트 단위 지도학습에는 음성 샘플(DY→ee)이 필요함.
  - eμ는 "학습 때 못 본 조합"이라 일반화 테스트로 중요. DY는 eμ를 직접 만들지 않으므로
    DY→ττ(→eμ), tt̄, WW→eμ 같은 샘플이 필요.

## 2. 사용 모델

- **주 모델: ParticleNet** — kNN 그래프 + EdgeConv. 국소 이웃 정보를 모음.
- **비교 모델: Particle Transformer (ParT)** — 모든 쌍에 attention. "국소 vs 전역" 비교가 목적.
  - 둘 다 원래 지도학습 분류용이라, 이상탐지로 갈 때는 같은 틀(autoencoder 등)로 감싸야 공정한 비교가 됨.
- **구조 방향 (사용자 결정)**: 2단계 계층 구조.
  1. 트랙 인코더: 트랙 하나(33점)를 벡터 하나(트랙 토큰)로 압축. 점이 트랙을 따라 순서가 있으므로
     순서 있는 시퀀스(1D conv, 작은 RNN/Transformer)가 순서 없는 집합(PointNet)보다 맞음.
  2. 이벤트 모델: 트랙 토큰 집합 + hit/segment 집합 → ParticleNet 또는 ParT.
  - 점 단위로 넣으면 이벤트당 수천 점이라 ParT의 O(N²) 메모리가 부담. 트랙 토큰 단위면 수십~수백 개.
- **궤적 표현 (사용자 결정)**: ISpy 방식(3차 Bezier, 33점)을 그대로 사용. helix로 바꾸지 않음.
  - 참고: Bezier는 양 끝점과 방향만 맞춘 보간이라 중간 점은 측정값이 아님.
    33점의 정보량은 끝점·방향 12개 수와 동일.

## 3. 데이터 위치 (lxplus)

| 항목 | 값 |
|---|---|
| 데이터셋 | `/ZToMuMu_M-50To120_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16RECO-106X_mcRun2_asymptotic_v13_ext1-v2/AODSIM` |
| 학습용 출력 | `/eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train/pointcloud/Run_1/*.npz` |
| 이벤트 수 | 20,000 요청 (`--nevents 20000 --nfiles 3 --per-ig 100 --no-html`) |
| 생성 로그 | `/eos/user/j/jieun/zmm_train.log`, 출력 폴더의 `logs/manifest.json` |
| 100개 테스트 출력 | `/eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_M-50To120_..._ext1-v2/` |
| GlobalTag | `106X_mcRun2_asymptotic_v13` (UL16 postVFP) |
| CMSSW | `/afs/cern.ch/user/j/jieun/CMSSW_10_6_30` (ISpy `cmssw10` 브랜치, GEM 패치 적용) |

**검증 완료 (2026-10-14)**: npz 20,000개 모두 읽힘, 오류 0, 중복 (run, lumi, event) 0, manifest 20,000 = 요청 20,000.
결과 JSON: `/eos/user/j/jieun/zmm_train_check.json`. 수치는 7절. 새 샘플도 같은 방법으로 확인
(lxplus, 깨끗한 셸, PointCloud_Agent 폴더에서):
```bash
grep -c "duplicate event" /eos/user/j/jieun/zmm_train.log        # 0 이어야 함
python3 -m ml.check_dataset /eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train --json /eos/user/j/jieun/zmm_train_check.json
```
`ml/check_dataset.py`는 모든 npz를 학습과 같은 경로(`ml/inputs.py`)로 읽어 npz 수·manifest 이벤트 수·
읽기 오류·중복 (run, lumi, event)·`has_muon` 비율·입력별 크기·`track_is_muon` 매칭 거리를 출력함.
오류나 중복이 있으면 종료 코드 2. `--max-events 1000`으로 일부만 먼저 볼 수 있음.

파일 이름은 `Event_<event>_LS<lumi>.npz` (MC는 run=1이라 lumi로 구분).

## 4. npz 스키마 (`agent/pointcloud.py`)

좌표 단위는 **m** (iSpy 관례). 각 배열의 열 이름은 `<키>_columns`에 같이 저장됨. 메타데이터: `run`, `event`, `lumi`.

| 키 | 열 | 점 정의 / 비고 |
|---|---|---|
| `tracks` | x, y, z, pt, track_id | trackExtrapolator, pT > 1 GeV. 트랙당 Bezier 33점 |
| `track_info` | pt, eta, phi, charge, chi2, ndof, track_id | 트랙당 1행, *i*행 = `tracks`의 track_id *i* |
| `elec_tracks` | x, y, z, pt, track_id | GSF 전자 (재구성 결과) |
| `muon_tracks` | x, y, z, pt, track_id | tracker muon (재구성 결과) |
| `global_muon` | x, y, z, pt, muon_id | `Points_V1` 실제 점, 뮤온마다 점 수 다름 (재구성 결과) |
| `standalone_muon` | x, y, z, pt, muon_id | Bezier (재구성 결과) |
| `muon_chambers` | x, y, z, detid | 챔버 상자 중심. **재구성 뮤온과 매칭된 챔버만** |
| `vertices`, `secondary_vertices` | x, y, z | |
| `ecal_eb`, `ecal_ee` | x, y, z, energy, eta, phi, time, detid | 결정 앞면 대각선 교점. E > 0.25 / 0.5 GeV |
| `hcal_hb`, `hcal_he`, `hcal_hf`, `hcal_ho` | 위와 같음 | 셀 중심(꼭짓점 8개 평균). E > 0 |
| `csc_segments` | x1, y1, z1, x2, y2, z2, endcap, station, ring, chamber, detid | 양 끝점 |
| `dt_segments` | x1, y1, z1, x2, y2, z2, wheel, station, sector, detid | 양 끝점 |
| `rpc_hits` | x, y, z, region, station, sector, layer, detid | u/v/w 교차의 중심 |

주의:
- `track_id`, `muon_id`는 각 키 안에서만 쓰는 번호. `tracks`와 `muon_tracks`를 잇는 대응 정보는 **없음**.
- dtype이 섞여 있음 (트랙·vertex float32, 칼로리미터·챔버·segment float64, 빈 배열 float32).
  `ml/inputs.py`가 모두 float32로 통일하고 detid는 버리므로 학습에는 문제 없음.
- npz에 없는 것: photons, PF jets, MET (재구성 결과라 입력으로 쓰면 안 됨), ES preshower.

## 5. 모델 입력 (`ml/inputs.py`)

numpy만 사용 (로컬에 torch 없음). torch Dataset 인터페이스와 같음.

```python
import glob
from ml.inputs import NpzEventDataset, collate
paths = glob.glob("/eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train/pointcloud/Run_*/*.npz")
ds = NpzEventDataset(paths, seed=0, pt_min=10.0,
                     hit_keys=("ecal_eb", "ecal_ee", "hcal_hb", "hcal_he"))
batch = collate([ds[i] for i in range(8)])     # torch: DataLoader(ds, collate_fn=collate)
```

| 배치 키 | 모양 | 내용 |
|---|---|---|
| `track_points` | (B, T, 33, 3) | 트랙별 점 x, y, z (안쪽→바깥쪽 순서 유지) |
| `track_feats` | (B, T, 1) | 트랙별 pT |
| `track_mask` | (B, T) | 실제 트랙 = True (나머지는 패딩) |
| `hit_feats` | (B, H, 5) | 칼로리미터 x, y, z, energy, time |
| `hit_det` | (B, H) | `hit_keys` 안의 위치 (기본: 0=EB 1=EE 2=HB 3=HE 4=HF 5=HO) |
| `hit_mask` | (B, H) | |
| `seg_feats` | (B, S, 6) | segment 양 끝점 |
| `seg_det` | (B, S) | 0=CSC, 1=DT |
| `seg_mask` | (B, S) | |
| `rpc_feats` | (B, R, 3) | RPC hit x, y, z |
| `rpc_mask` | (B, R) | |
| `has_muon` | (B,) | 라벨: pT > `pt_min`인 global muon이 1개 이상 |
| `n_muons` | (B,) | 그 개수 |
| `track_is_muon` | (B, T) | 트랙 단위 라벨: pT > `pt_min`인 tracker muon과 끝점이 일치하는 트랙 (패딩은 False) |

처리 원칙 (모델이 번호를 학습하지 않도록):
- `track_id`는 33점을 묶는 데만 쓰고 버림. 트랙 순서는 읽을 때마다 무작위로 섞음.
- hit, segment, RPC도 `detid`와 챔버 번호를 버리고 순서를 섞음.
- `event(path, rng=None)`이면 섞지 않음 (검증·디버깅용).
- 새 키가 없는 옛 npz도 빈 배열로 처리됨.

## 6. 라벨과 정답 누출 (중요한 결정)

- **입력에 절대 넣지 않는 것**: `muon_tracks`, `global_muon`, `standalone_muon`, `elec_tracks`, `muon_chambers`.
  - 재구성 결과라서 사실상 정답. 학습 때 넣고 테스트 때 빼면 모델이 의지하던 입력이 사라져 실패함.
  - `muon_chambers`는 재구성 뮤온과 매칭된 챔버만 기록되므로 역시 정답이 새는 입력.
- **정당한 뮤온 시스템 입력**: CSC/DT segment, RPC hit. 뮤온 재구성 이전의 국소 측정값.
  - 노이즈·pileup으로 뮤온 없는 이벤트에도 일부 생김 → 모델은 "segment가 있는가"가 아니라
    "트랙 연장선 위에 segment가 있는가"를 배워야 함.
  - ee 샘플을 만들면 segment 개수 분포를 μμ와 비교할 것 (차이가 크면 개수만 보는 지름길 위험).
- **라벨**: 샘플 이름이 아니라 재구성 뮤온 기준 (`global_muon` pT > 10 GeV). ee 샘플에도 heavy flavor·pileup
  뮤온이 섞일 수 있고, μμ 샘플에도 뮤온이 acceptance 밖으로 나가 라벨 0인 이벤트가 있음.
- **트랙 단위 라벨 `track_is_muon`** (`ml/inputs.py`, npz 재생성 불필요): `tracks`와 `muon_tracks`는 둘 다
  Extras_V1의 pos_1/pos_2를 지나는 Bezier라서, 같은 트랙이면 양 끝점이 일치함. 두 끝점 거리 중 큰 값이
  `match_tol`(기본 1e-3 m, 실행 시 선택) 미만이면 매칭. 장점: μμ 샘플 하나로 학습 가능, 샘플 간 차이 문제 없음.
  - **UL16 2만 개에서 확인**: pT > 10 tracker muon 26,372개 중 25,899개가 거리 ≤ 1e-6 m (사실상 0),
    1e-6~1e-3 m는 2개뿐 → 1e-3 m 기준은 간격이 뚜렷해 적절. 매칭 안 된 471개(1.8%)는 대부분 0.1 m 이상 떨어져
    있어 해당 트랙이 `tracks`(quality 선택 + pT > 1 GeV)에 없는 경우로 보임.
  - 라벨 트랙 25,900개 / 전체 트랙 약 208만 개 = **약 1.2%** → 학습 시 클래스 불균형 처리, 평가는 ROC/PR.
  - 기준은 tracker muon, 이벤트 라벨 `has_muon`은 global muon이라 개수가 다를 수 있음 (check_dataset이 셈).

## 7. 측정된 데이터 통계 (UL16 Z→μμ)

| 항목 | 값 | 출처 |
|---|---|---|
출처: 2만 개 전체 `ml/check_dataset.py` (좌표 검증만 100개 테스트).

| 항목 | 값 |
|---|---|
| `has_muon` (global muon pT > 10) | 16,223/20,000 = **0.811**. n_muons 0: 3,777 / 1: 6,911 / 2: 9,258 / 3: 54 |
| 트랙 수 | 이벤트당 평균 104.1, 중앙값 98, p99 241, 최대 391 |
| 칼로리미터 hit 전체 (6종) | 평균 353.4, p99 858, 최대 1,452 |
| `ecal_eb` / `ecal_ee` | 이벤트당 평균 20.7 / 117.4 |
| `hcal_hb` / `hcal_he` / `hcal_hf` / `hcal_ho` | 이벤트당 평균 0.2 / 1.3 / 213.8 / 0.0 |
| `csc_segments` | 19,474 이벤트에 있음, 평균 8.7, p99 26, 최대 54 |
| `dt_segments` | 15,800 이벤트, 평균 3.7, p99 16, 최대 87 |
| `rpc_hits` | 19,964 이벤트, 평균 10.2, p99 22, 최대 33 |
| tracker muon (pT > 10) | 26,372개, 그중 `tracks`와 매칭 25,901개 (98.2%) |
| 트랙 라벨 수 ≠ global muon 수인 이벤트 | 516 (2.6%) |
| 좌표 검증 | CSC \|z\| 5.80~10.46, DT r 4.14, HF \|z\| ≈ 12.0, RPC r 3.56~7.08 (m) | 검출기 위치와 일치 |

## 8. 물리적 한계 (논문에 명시할 것)

- **칼로리미터 hit은 reco 선택의 부분집합**: AOD의 `reducedEcalRecHits*`, `reducedHcalRecHits`는 재구성 객체
  주변만 남김. 그래서 HB/HE/HO가 거의 비어 있음. reco가 객체로 못 만든 에너지 퇴적은 애초에 빠져 있음.
- **tracks는 generalTracks 전체가 아님**: trackExtrapolator(quality 선택) + pT > 1 GeV.
- **같은 입자가 여러 번**: 전자 GSF 트랙과 tracker muon 트랙이 `tracks`에도 들어 있을 수 있음.
- **트랙당 33점 고정**: 점 개수 기준으로는 저 pT 트랙 비중이 과대. 트랙 토큰 구조로 완화됨.
- **HF는 대부분 전방 pileup/underlying event**: 뮤온(|η| < 2.4)과 무관. 샘플 간 pileup이 다르면 지름길 위험.

## 9. 학습 계획 (합의된 내용)

- **이벤트 수**: 클래스당 2만 개. 학습/검증/테스트 70/15/15. 학습 데이터 10%·30%·100%로 learning curve를
  그려 포화 여부 확인, 계속 오르면 5만~10만으로 확대.
- **칼로리미터 비교 실험** (`hit_keys`로 실행 시 선택):
  | 설정 | `hit_keys` |
  |---|---|
  | A | `("ecal_eb", "ecal_ee")` |
  | B (권장 시작점) | `("ecal_eb", "ecal_ee", "hcal_hb", "hcal_he")` |
  | C | B + `"hcal_hf"` |
  C가 B보다 크게 좋으면 뮤온이 아니라 pileup/전방 활동 차이를 배운 것 아닌지 의심.
- **음성 샘플**: 같은 UL16 캠페인(`RunIISummer20UL16RECO-106X_mcRun2_asymptotic_v13*`)의 DY→ee.
  캠페인·GlobalTag가 다르면 모델이 조건 차이를 배울 수 있음. 디스크 복제본 확인 필수
  (테이프 전용 샘플이 많음: `dasgoclient -query="site dataset=..."`에서 `_Disk`나 `T2_*` 필요).
- **체크포인트에 같이 저장할 것**: 가중치, 모델 설정, 입력 설정(`hit_keys`, `pt_min`), 정규화 값(평균·표준편차),
  데이터 분할 목록, git commit, epoch별 지표, (이어 학습 시) optimizer 상태.
  저장 위치는 EOS (`/eos/user/j/jieun/...`), 실험마다 폴더 분리.

## 10. 환경

| 곳 | 상태 |
|---|---|
| 로컬 (WSL) | Python 3.12, numpy 2.5, **torch 없음**. GitHub SSH 키 없음 → push는 사용자가 직접 |
| lxplus | Python 3.9, numpy 2.0.2, plotly 설치됨. **AFS 홈 2GB 거의 가득** → 큰 파일은 EOS에 |
| lxplus 셸 | 에이전트·pip은 `cmsenv` 안 한 깨끗한 셸에서 (섞이면 `GLIBCXX` 오류) |
| 학습 장소 | **lxplus HTCondor GPU** (데이터가 EOS에 있음). 짧은 확인은 `lxplus-gpu.cern.ch`. 로그인 노드에서 장시간 학습 금지 |

## 11. 데이터 생성 명령 (참고)

```bash
# lxplus, 깨끗한 셸, voms-proxy-init 후. tmux 안에서 실행 권장.
cd /afs/cern.ch/user/j/jieun/CMSSW_10_6_30/src/CMS_iSpy_Analysis/PointCloud_Agent
python3 run_agent.py <DAS dataset> --name <출력 이름> --nevents 20000 --nfiles 3 --per-ig 100 --no-html
# 같은 데이터셋의 다음 배치: --first-file 3 --name <다른 이름>
# ig는 그대로 두고 npz만 다시 만들기: --steps pointcloud
```
- HTML은 `--html` / `--no-html`로 실행 시 선택 (기본 `--html`).
- UL16 Z→μμ 파일당 평균 약 11,700 이벤트 (332개 파일, 3,893,165 이벤트).
- 이벤트당 대략 cmsRun 0.05초, 점구름 0.1초, 용량 0.5MB.

## 12. 사용자 작업 방식

- 설정은 코드 파일을 고치지 않고 **실행할 때 옵션으로** 고를 수 있게 만들 것 (예: `--html/--no-html`, `hit_keys=`).
- 대화는 한국어.
- 커밋은 사용자가 요청할 때. push는 로컬에 SSH 키가 없어 사용자가 직접 함.

## 13. 남은 일

1. ~~2만 개 생성 결과 검증~~ 완료 (3절, 7절).
2. ~~학습 장소 결정~~ lxplus HTCondor GPU (14절).
3. 둘 중 하나로 시작:
   - DY→ee 샘플 찾기·생성 → 이벤트 단위 학습 (μμ vs ee).
   - 트랙 단위 학습 (μμ만으로 가능). 라벨 `track_is_muon`은 준비됨 (UL16에서 매칭 확인 필요).
4. 정규화 방식 결정 (좌표 m 단위, 에너지·pT는 로그 스케일 등).
5. ~~ParticleNet 구현~~ 코드 완료 (14절), lxplus에서 첫 학습 필요. 이어서 ParT 비교.
6. eμ 테스트 샘플 확보.

## 14. 학습 코드 (ParticleNet)

로컬에서는 옛 npz와 임의로 만든 입력으로 **실행만** 확인함 (CPU torch). 실제 수치는 lxplus에서.

| 파일 | 역할 |
|---|---|
| `ml/pack.py` | npz → train/val/test 묶음 (`<pack>/<split>/*.bin` + `meta.json`, `pack.json`). numpy만, Python 3.9 |
| `ml/packed.py` | 묶음 읽기, 특징 변환, 정규화(train 합계에서), 패딩 |
| `ml/particlenet.py` | 트랙 인코더(`conv` 1D Conv / `mlp` 끝점·방향 12개) → 노드 종류 임베딩 → EdgeConv 3블록 → 트랙별/이벤트별 출력 |
| `ml/train.py` | 학습·검증·테스트, 체크포인트, 지표 |
| `ml/condor/pack.sub`, `pack.sh` | condor CPU 작업으로 묶기 (scratch에 쓰고 xrdcp로 EOS에 복사) |
| `ml/condor/train.sub`, `train.sh` | condor GPU 작업으로 학습 (묶음을 scratch로 복사, 결과는 EOS에 바로 씀) |

설계:
- **묶음은 모든 것을 담고, 실험 설정은 학습 때 옵션으로 고름.** 트랙 변수 6개 전부, 칼로리미터 6종 전부,
  트랙마다 가장 가까운 tracker muon까지 거리와 그 pT, 이벤트마다 global muon pT (최대 4개)를 저장.
  `--hit-keys`, `--track-features`, `--pt-min`, `--match-tol`, `--segments/--no-segments`, `--rpc/--no-rpc`를 바꿔도 다시 묶을 필요 없음.
- 정규화: train 분할에서 칼로리미터 종류별로 합을 저장, 학습 때 고른 `hit_keys`만 합쳐 평균·표준편차 계산.
- 변환: pT → log, φ → sin/cos, χ² · ndof → log1p, 에너지 → log, segment → 중점 + 단위 방향.
- 첫 EdgeConv의 kNN은 (η, φ)에서 φ를 2π로 감아 계산 (트랙은 바깥 끝점, 나머지는 hit 위치).
  이후 블록은 학습된 특징 공간에서. 패딩 노드는 이웃이 되지 않음.
- `--task track` (기본): 트랙마다 `track_is_muon`. `--task event`: `has_muon`.
- 검증 지표: AUC, AP. track task는 pT > `pt_min` 트랙만의 `auc_pt`/`ap_pt`도 계산하고, best는 `auc_pt`로 고름
  (pT < `pt_min` 트랙은 정의상 음성이라 전체 AUC는 과대평가됨).
- 체크포인트(`last.pt`, `best.pt`): 가중치, optimizer·scheduler·scaler 상태, 옵션 전체, 입력 설정, 정규화 값,
  묶음 정보, git commit, epoch별 지표, 분할 목록. 같은 `name`으로 다시 제출하면 `last.pt`에서 이어 학습.

실행 (lxplus, PointCloud_Agent 폴더):
```bash
# 1) 묶기 (한 번)
condor_submit ml/condor/pack.sub targets=/eos/user/j/jieun/PointCloud_Agent_output/ZToMuMu_UL16_train pack=/eos/user/j/jieun/ml/packed/zmm_ul16_v1
# 2) 학습: 설정마다 name 하나
condor_submit ml/condor/train.sub pack=/eos/user/j/jieun/ml/packed/zmm_ul16_v1 runs=/eos/user/j/jieun/ml/runs name=B_conv
condor_submit ml/condor/train.sub pack=... runs=... name=B_mlp "args=--track-encoder mlp"
condor_submit ml/condor/train.sub pack=... runs=... name=A_conv "args=--hit-keys ecal_eb ecal_ee"
```
- LCG view 기본값: 묶기·학습 모두 `LCG_107_cuda` (`x86_64-el9-gcc11-opt`, Python 3.11.9, torch 2.5.1, numpy 1.26.4).
  lxplus에서 확인된 cuda view: `LCG_105_cuda` (3.9.12, torch 2.1.0a0), `LCG_106_cuda` (3.11.9, torch 2.3.1), `LCG_107_cuda`.
  다른 view는 `lcg=<폴더>`로 고름. 코드는 torch 2.1(Python 3.9, numpy 1.23)과 2.14에서 로컬 실행 확인.
- 로그: `ml/condor/logs/` (AFS, 작음). 결과: `<runs>/<name>/` (EOS).
- GPU 메모리가 부족하면 `"args=--batch-size 16"`.
