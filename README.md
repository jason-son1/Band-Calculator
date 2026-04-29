# 🔬 Band Dispersion Visualizer

> 위상물리/응집물질물리 연구자를 위한 **2×2 해밀토니안 밴드 구조 시각화** 및 실시간 파라미터 튜닝 도구

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Features

- **수식 기반 해밀토니안 입력**: 파울리 행렬 기저 `d₀σ₀ + dₓσₓ + dᵧσᵧ + d_zσ_z`
- **동적 파라미터 추출**: 수식에서 자유 파라미터를 자동 인식 → 슬라이더 생성
- **인터랙티브 시각화**: 1D 밴드 구조 + 3D 에너지 표면 (Plotly)
- **프리셋 모델**: QWZ, SSH, Graphene, Haldane 내장
- **커스텀 k-경로**: 고대칭점 좌표 직접 입력 가능

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI Framework | Streamlit |
| Symbolic Math | SymPy |
| Numerical Computing | NumPy, SciPy |
| Visualization | Plotly |

## Project Structure

```
Band-Calculator/
├── app.py                  # Streamlit 메인 앱
├── requirements.txt        # 의존성
├── .streamlit/config.toml  # 테마 설정
└── core/
    ├── parser.py           # SymPy 수식 파싱
    ├── band_calculator.py  # 고유값 계산
    ├── visualizer.py       # Plotly 시각화
    ├── presets.py          # 프리셋 모델
    └── utils.py            # k-경로 유틸리티
```

## License

MIT
