# Raman Spectroscopy Processor — 사용 설명서 (v2.4)

본 문서는 Raman Processor의 **상세 사용 설명서**입니다. 설치부터 GUI 각 패널 사용법, 배치 파이프라인, 다변량 분석, 하이퍼스펙트럴 매핑, 플러그인 작성, CLI 사용까지 전 기능을 다룹니다.

---

## 목차

1. [설치 및 실행](#1-설치-및-실행)
2. [메인 화면 구성](#2-메인-화면-구성)
3. [데이터 파일 포맷](#3-데이터-파일-포맷)
4. [단일 스펙트럼 처리 워크플로](#4-단일-스펙트럼-처리-워크플로)
5. [배치 처리 (Batch Processing)](#5-배치-처리-batch-processing)
6. [피크 검출 및 피팅](#6-피크-검출-및-피팅)
7. [PCA 주성분 분석](#7-pca-주성분-분석)
8. [NMF 비음수 행렬 분해](#8-nmf-비음수-행렬-분해)
9. [MCR-ALS](#9-mcr-als)
10. [클러스터링](#10-클러스터링)
11. [파장 보정 (Wavelength Calibration)](#11-파장-보정-wavelength-calibration)
12. [하이퍼스펙트럴 매핑](#12-하이퍼스펙트럴-매핑)
13. [플러그인 작성](#13-플러그인-작성)
14. [CLI (Headless) 사용](#14-cli-headless-사용)
15. [파라미터 저장/불러오기](#15-파라미터-저장불러오기)
16. [문제 해결 FAQ](#16-문제-해결-faq)

---

## 1. 설치 및 실행

### 요구 사항
- Python **3.11 이상**
- Windows / macOS / Linux

### 설치

```bash
git clone https://github.com/GTAEKIM/Raman-processor.git
cd Raman-processor

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
# 선택: UMAP 시각화가 필요하면
pip install umap-learn
```

### 실행

```bash
# GUI
python main_app.py

# CLI (도움말)
python cli.py --help
```

---

## 2. 메인 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│  [Import Data] [Import Parameters] [Hyperspectral Mapping]  │
├──────────────────┬──────────────────────────────────────────┤
│ 좌측 컨트롤 패널 │                                          │
│  1. Data Load    │                                          │
│  2. Pre-process  │         중앙: 스펙트럼 플롯              │
│  3. Smoothing    │         (Matplotlib, 확대/팬 가능)       │
│  4. Baseline     │                                          │
│  5. Normalize    │                                          │
│  6. Export       │──────────────────────────────────────────│
│  7. Batch / PCA  │ [Raw] [Smooth] [Baseline] [Final] 체크   │
└──────────────────┴──────────────────────────────────────────┘
```

- **툴바**: 파일 불러오기, 파라미터 불러오기/저장, 하이퍼스펙트럴 매핑 창 열기.
- **좌측 패널**: 파이프라인 단계별 파라미터 입력.
- **중앙 플롯**: 현재 선택된 샘플의 스펙트럼을 단계별로 겹쳐서 표시.
- **하단 체크박스**: 어떤 곡선을 보일지 선택 (Raw / Smoothed / Baseline / Final).

---

## 3. 데이터 파일 포맷

### (A) Multi-spectrum 와이드 테이블
가장 일반적인 형식입니다.

|            | 400.0 | 401.0 | 402.0 | ... |
|------------|-------|-------|-------|-----|
| Sample 1   | 123   | 125   | 120   | ... |
| Sample 2   | 110   | 115   | 118   | ... |

- **1행**: Raman shift (cm⁻¹)
- **1열**: 샘플명
- **나머지 셀**: 강도값

### (B) 2열 단일 스펙트럼
`.txt`, `.asc`, `.dat` 등에서 흔히 쓰이는 포맷. 숫자 2열만 있으면 자동 인식합니다.

```
400.0   123.4
401.0   125.1
...
```
→ 파일명이 샘플명으로 사용됩니다.

### (C) 하이퍼스펙트럴 매핑
첫 두 열이 `X`, `Y` 좌표, 나머지 열이 Raman shift인 테이블.

| X   | Y   | 400.0 | 401.0 | ... |
|-----|-----|-------|-------|-----|
| 0.0 | 0.0 | 123   | 125   | ... |
| 0.5 | 0.0 | 119   | 121   | ... |

→ **Hyperspectral Mapping** 창에서 불러와야 합니다 ([§12](#12-하이퍼스펙트럴-매핑)).

> `#` 또는 `%`로 시작하는 주석 줄은 자동으로 건너뜁니다.

---

## 4. 단일 스펙트럼 처리 워크플로

### 4.1 Data Load
- **[Import Data]** 버튼으로 파일 선택.
- 샘플 리스트에서 처리할 스펙트럼을 선택.
- **Raman shift range**에 lower / upper bound 입력 후 **Apply Range**.

### 4.2 Pre-processing
- **Cosmic ray removal**: 체크 시 MAD 기반 스파이크를 제거.
- 제거 강도는 Z-score threshold와 window로 조절 가능.

### 4.3 Smoothing (Savitzky-Golay)
- **Polynomial order**: 보통 2~3.
- **Window size** (홀수): 5, 11, 15, 21, ... (노이즈가 심하면 크게).
- 실시간으로 플롯에 반영됩니다.

### 4.4 Baseline Correction
- 알고리즘 선택 (airPLS / arPLS / ALS / SNIP / ATQ / STQ / AH / SH / Morphological / 플러그인).
- **[Interactive Baseline Adjust...]**: 슬라이더로 파라미터를 돌려가며 실시간 프리뷰.
- 대표 파라미터:
  - airPLS/arPLS/ALS: `λ`(lam) 클수록 매끄러움, `diff_order` 2 권장.
  - SNIP: `max_half_window` (피크 폭의 2~3배 정도로).
  - Morphological: `half_window` (피크 폭보다 크게).

### 4.5 Normalization
- **SNV** (표준정규화): 샘플 간 정량 비교에 가장 많이 씀.
- **Vector / Area / MinMax / MaxPeak**: 용도에 따라.
- 정규화를 적용하면 Raw/Smoothed/Baseline 그래프는 자동으로 숨겨집니다.

### 4.6 Derivative (선택)
- 1차 또는 2차 도함수 (SG 기반) 계산. 오버래핑된 피크를 분리할 때 유용.

### 4.7 Export
- **[Export Data]**: xlsx/csv로 저장. 파라미터 JSON이 동일 경로에 함께 저장됩니다.

---

## 5. 배치 처리 (Batch Processing)

여러 스펙트럼을 동일한 파이프라인으로 한 번에 처리합니다.

1. 위 §4처럼 파이프라인 파라미터를 결정합니다.
2. **Batch / PCA** 섹션에서:
   - **Parallel**: 다중 코어 사용 (joblib).
   - **n_jobs**: `-1`이면 전 코어 사용.
3. **[Run Batch]** 클릭.
4. 완료 후:
   - 진행률 바와 요약 (processed / failed / QC flagged).
   - 결과는 내부에 저장되어 PCA/NMF/클러스터링/MCR-ALS 버튼으로 바로 이어서 분석 가능.
5. **[Export Batch Result]**: 다중 시트 Excel 출력
   - `Processed` — 처리된 스펙트럼
   - `QC` — SNR, saturation, spike count, baseline drift, flag
   - (PCA/NMF를 돌렸다면) 결과 시트 자동 포함.

### QC 지표
- **SNR**: 신호대잡음비.
- **Saturation**: 포화 비율.
- **Spike count**: 검출된 코스믹 레이 개수.
- **Baseline drift**: 베이스라인 추세 크기.
- **Flag**: 자동 품질 플래그 (낮은 SNR / 높은 포화 / 과도한 drift).

---

## 6. 피크 검출 및 피팅

**Peak Analysis** 창.

### 자동 검출
- `prominence`, `height`, `distance` 임계치 설정.
- 검출된 피크는 플롯에 마커로 표시.

### 피팅 (lmfit)
- 모델 선택: **Gaussian / Lorentzian / Voigt / PseudoVoigt**.
- 각 피크별 초기값은 검출 결과로부터 자동 세팅.
- 결과 테이블: center, amplitude, FWHM, area, R².
- 결과 Excel로 내보내기 가능.

> **팁**: 좁고 대칭인 피크는 Gaussian, 길게 끌리는 꼬리가 있으면 Lorentzian, 혼합이면 Voigt / PseudoVoigt.

---

## 7. PCA 주성분 분석

배치 후 **[PCA]** 버튼 → PCA 결과 창.

### 옵션
- **n_components**: 기본 10 (표본 수보다 작게).
- **Scaling**: `auto` / `mean` (센터링만) / `pareto` (SD의 √) / `none`.

### 결과 패널
- **Scores plot** (PC1 vs PC2 등) — 각 샘플의 투영.
- **Loadings plot** — 각 PC의 파장별 기여도.
- **Scree / 누적 설명 분산** — 적절한 PC 개수 판단.
- **Hotelling's T²** — 모델 내부 이상치 지표. 95% / 99% 한계선 표시.
- **Q-residuals** — 모델 외부 이상치 지표. Jackson-Mudholkar 한계.

> T²와 Q가 모두 높은 샘플은 이상치 가능성 큼.

### Export
- Scores / Loadings / Variance를 Excel로.

---

## 8. NMF 비음수 행렬 분해

배치 후 **[NMF]** → 구성요소 개수(K) 입력.

- **초기화**: NNDSVDA (결정론적, 빠른 수렴).
- **Components**: 각 컴포넌트의 스펙트럼 (순수성분 추정).
- **Weights**: 각 샘플에서 해당 컴포넌트의 기여도.
- **Reconstruction error**: 낮을수록 K가 충분.

> K는 보통 2~5에서 시작하여 reconstruction error가 급격히 감소하지 않는 지점을 선택.

---

## 9. MCR-ALS

**Multivariate Curve Resolution – Alternating Least Squares** (pymcr 기반).

- NMF보다 제약조건 (비음수 / 단조성 / 폐쇄성)을 더 유연하게 걸 수 있음.
- K (순수 성분 수) 추정 → 초기 스펙트럼 자동/수동 선택 → 반복 최적화.
- 결과: `Spectra`(순수 성분), `Concentrations`(샘플별 농도), 수렴 로그.
- 물리적으로 해석 가능한 spectral decomposition이 목적일 때 권장.

---

## 10. 클러스터링

**Clustering** 창. 세 가지 방법 지원.

### HCA (Hierarchical)
- Linkage: ward / complete / average.
- **Dendrogram** 제공 → 클러스터 수(k) 결정 후 `fcluster`로 레이블링.

### K-means
- 사용자가 k 지정.
- **Silhouette score**로 품질 평가 (−1~1, 높을수록 좋음).

### UMAP (선택)
- `umap-learn` 설치 시 활성화.
- 2D 임베딩 시각화. 클러스터 구조를 눈으로 빠르게 파악.

각 샘플에 부여된 cluster 레이블은 Excel로 내보낼 수 있습니다.

---

## 11. 파장 보정 (Wavelength Calibration)

표준 물질의 알려진 피크 위치로 x축을 재보정합니다.

### 내장 레퍼런스
| 물질 | 주요 피크 (cm⁻¹) |
|---|---|
| Silicon | 520.7 |
| Polystyrene | 621, 1001, 1031, 1602, 3054 |
| Cyclohexane | 801, 1028, 1266, 1444, 2852, 2923 |
| Acetonitrile | 379, 918, 2249, 2942 |
| Ethanol | 883, 1052, 1095, 1454, 2875, 2930 |

### 절차
1. 레퍼런스 물질 스펙트럼 선택.
2. 물질 선택 → 표준 피크 자동 로드.
3. 관측된 피크 위치를 각 레퍼런스 피크에 대응시킴 (수동 클릭 or 자동 매칭).
4. 다항식 차수 (1~3) 선택 → 피팅.
5. **[Apply Calibration]** → 전체 샘플의 x축 재매핑.

### 결과
- 피팅 잔차(residual) 그래프와 각 피크의 보정 전후 차이 출력.

---

## 12. 하이퍼스펙트럴 매핑

**Hyperspectral Mapping** 창 (툴바에서 열기).

### 12.1 파일 불러오기
- `X, Y, shift1, shift2, ...` 형식의 테이블 (§3 Type C).
- 3D 큐브 `[Y × X × shift]`로 자동 변환.

### 12.2 밴드 적분 히트맵
- Low / High (cm⁻¹) 입력.
- Method:
  - **trapezoid** — 사다리꼴 적분 (기본).
  - **sum** — 단순 합.
  - **max** — 밴드 내 최대값.
  - **mean** — 평균.
- **[Compute Map]** → 2D 히트맵 생성. colormap 변경 가능.

### 12.3 픽셀 클릭
- 히트맵에서 임의의 픽셀을 클릭 → 우측에 해당 픽셀의 **스펙트럼** 표시.
- 선택된 밴드는 주황색 영역으로 하이라이트.

### 12.4 배치 파이프라인으로 전송
- **[Send Cube to Batch Pipeline]** → 모든 픽셀 스펙트럼을 DataFrame으로 변환해 메인 창으로 전달.
- 이후 PCA / NMF / MCR-ALS / Clustering 등을 메인 창에서 그대로 실행 가능.

### 12.5 내보내기
- **[Export Map to Excel...]**: 히트맵 배열을 xlsx/csv로.

---

## 13. 플러그인 작성

`plugins/baseline/` 디렉터리에 `.py` 파일을 넣기만 하면 앱 시작 시 자동 등록됩니다.

### 최소 구조

```python
# plugins/baseline/my_algo.py
import numpy as np

def _my_baseline(x, y, params):
    """x, y: np.ndarray / params: dict. 같은 길이의 baseline 반환."""
    strength = float(params.get("strength", 1.0))
    # ... 계산 ...
    baseline = np.zeros_like(y)  # 예시
    return baseline

def register(registry):
    registry["register"](
        short_code="myalgo",
        display_name="My Custom Baseline",
        compute_fn=_my_baseline,
        default_params={"strength": 1.0},
    )
```

- `short_code`: 내부 키 (CLI `--baseline myalgo` / JSON에서 사용).
- `display_name`: GUI에 노출될 이름.
- `default_params`: 기본 파라미터 dict.

### 적용 경로
1. GUI 실행 시 자동 등록 → 베이스라인 드롭다운에 나타남.
2. CLI에서도 `--baseline <short_code>`로 선택 가능.
3. 파라미터 JSON에서 `{"baseline": {"algorithm": "myalgo", "params": {...}}}` 지정 가능.

샘플은 `plugins/baseline/rolling_ball.py` 참고.

---

## 14. CLI (Headless) 사용

GUI 없이 전체 파이프라인을 실행합니다.

### 도움말
```bash
python cli.py --help
```

### 기본 사용
```bash
python cli.py input.xlsx output.xlsx
```

### 전체 옵션 예시
```bash
python cli.py input.xlsx output.xlsx \
    --range 400 3300 \
    --sg-order 2 --sg-window 15 \
    --baseline airpls \
    --normalize snv \
    --cosmic \
    --parallel --n-jobs -1 \
    --pca --pca-components 10 --pca-scaling auto \
    --nmf 3
```

### JSON 파라미터 오버라이드
메인 앱에서 저장한 파라미터 JSON을 그대로 사용 가능:
```bash
python cli.py input.xlsx output.xlsx --params my_params.json
```
(이 경우 플래그는 무시됩니다.)

### 출력
- `output.xlsx`: 다중 시트 엑셀
  - `Processed` — 처리된 스펙트럼.
  - `QC` — 품질 지표.
  - `PCA_Scores` / `PCA_Variance` — `--pca` 시.
  - `NMF_Weights` / `NMF_Components` — `--nmf K` 시.
- `output.json`: 사용한 파라미터 전체 (재현성 보장).

### 주요 플래그 요약
| 플래그 | 의미 |
|---|---|
| `--range LOW HIGH` | Raman shift 범위 (기본 400 3300) |
| `--sg-order` / `--sg-window` | SG 파라미터 |
| `--baseline` | airpls, arpls, asls, snip, atq, stq, ah, sh, mor, 플러그인 short_code |
| `--normalize` | none, snv, vector, area, minmax, maxpeak |
| `--cosmic` | 코스믹 레이 제거 |
| `--parallel --n-jobs N` | 병렬 처리 |
| `--pca` / `--pca-components N` / `--pca-scaling` | PCA 수행 |
| `--nmf K` | K 성분 NMF |
| `--plugin-dir PATH` | 플러그인 디렉터리 (기본 `./plugins`) |
| `-v` | 디버그 로그 |

---

## 15. 파라미터 저장/불러오기

### 저장
- 단일 스펙트럼을 내보내면 `output.json`이 함께 생성됩니다.
- CLI도 동일하게 사이드카 JSON을 생성.

### 불러오기 (GUI)
- 툴바 **[Import Parameters]**로 JSON 선택 → 모든 파이프라인 설정이 UI에 반영됨.

### JSON 구조 (예)
```json
{
  "range": {"lower_bound": 400, "upper_bound": 3300},
  "smoothing": {"sg_poly_order": 2, "sg_frame_window": 15},
  "baseline": {
    "algorithm": "airpls",
    "params": {"lam": 1e6, "diff_order": 2}
  },
  "normalization": "snv",
  "preprocessing": {"apply_cosmic_ray": true},
  "derivative": {"enabled": false, "order": 1, "window": 15, "polyorder": 3}
}
```

---

## 16. 문제 해결 FAQ

**Q. `ModuleNotFoundError: pymcr` / `lmfit`**
A. `pip install -r requirements.txt`를 다시 실행하세요. 가상환경이 활성화되어 있는지 확인.

**Q. UMAP 버튼이 비활성화 상태입니다.**
A. `pip install umap-learn`으로 선택 의존성을 설치하고 앱을 재시작하세요.

**Q. 배치 처리에서 parallel 모드가 느립니다.**
A. 스펙트럼이 수십 개 이하이면 직렬이 더 빠를 수 있습니다. 병렬의 이득은 수백 개 이상부터 본격적으로 나타납니다.

**Q. PCA에서 T², Q의 한계선이 NaN으로 나옵니다.**
A. `n_components`가 샘플 수와 같으면 자유도가 0이라 F-분포 한계를 계산할 수 없습니다. n_components를 줄이세요.

**Q. 하이퍼스펙트럴 파일을 열었는데 "no X/Y columns" 에러.**
A. 첫 두 열의 헤더가 정확히 `X`, `Y`인지 확인 (대소문자 무관). 그 외 열 헤더는 숫자(Raman shift)여야 합니다.

**Q. 베이스라인 보정 후 결과가 이상합니다 (피크가 음수).**
A. airPLS/arPLS의 `λ` 값을 키우거나 SNIP의 `max_half_window`를 피크 폭의 2~3배 정도로 조정하세요. Interactive 창에서 슬라이더로 확인 가능합니다.

**Q. 플러그인이 로드되지 않습니다.**
A. `plugins/baseline/` 경로가 맞는지, `register(registry)` 함수가 최상위에 정의되어 있는지, import 에러가 없는지 콘솔 로그를 확인하세요.

**Q. CLI에서 `Permission denied` 로 output 파일을 못 씁니다.**
A. 해당 Excel 파일이 열려있지 않은지 확인하세요 (Windows는 열린 파일에 쓸 수 없음).

---

## 라이선스

MIT License.

문의: GitHub Issues — https://github.com/GTAEKIM/Raman-processor/issues
