"""
Preset Hamiltonian models for common topological systems.
"""

PRESETS = {
    "Custom (직접 입력)": {
        "d0": "0",
        "dx": "",
        "dy": "",
        "dz": "",
        "params": {},
        "description": "직접 수식을 입력하세요.",
        "k_path": "square",
    },
    "QWZ (Qi-Wu-Zhang)": {
        "d0": "0",
        "dx": "sin(k_x)",
        "dy": "sin(k_y)",
        "dz": "m - cos(k_x) - cos(k_y)",
        "params": {"m": {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}},
        "description": "2D Chern insulator. m=0, ±2에서 위상 전이 발생.",
        "k_path": "square",
    },
    "SSH (Su-Schrieffer-Heeger)": {
        "d0": "0",
        "dx": "t1 + t2 * cos(k_x)",
        "dy": "t2 * sin(k_x)",
        "dz": "0",
        "params": {
            "t1": {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
            "t2": {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.01},
        },
        "description": "1D SSH 모델. t1=t2에서 위상 전이.",
        "k_path": "1d",
    },
    "Graphene (Low Energy)": {
        "d0": "0",
        "dx": "1 + cos(k_x) + cos(k_y)",
        "dy": "sin(k_x) + sin(k_y)",
        "dz": "Delta",
        "params": {"Delta": {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}},
        "description": "그래핀 저에너지. Delta=0이면 디랙 콘 형성.",
        "k_path": "hexagonal",
    },
    "Haldane Model": {
        "d0": "2*t2*cos(phi)*(cos(k_x)+cos(k_y)+cos(k_x-k_y))",
        "dx": "t1*(1+cos(k_x)+cos(k_y))",
        "dy": "t1*(sin(k_x)+sin(k_y))",
        "dz": "M_mass-2*t2*sin(phi)*(sin(k_x)-sin(k_y)-sin(k_x-k_y))",
        "params": {
            "t1": {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
            "t2": {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
            "M_mass": {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01},
            "phi": {"default": 1.57, "min": -3.14, "max": 3.14, "step": 0.01},
        },
        "description": "Haldane 모델. 양자 비정상 홀 효과(QAHE)의 원형 모델.",
        "k_path": "hexagonal",
    },
}
