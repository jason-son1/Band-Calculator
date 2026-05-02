"""
Hopping ↔ 수식 간 색상 매핑 관리자.

다이어그램의 Edge 색상과 해밀토니안 LaTeX 수식의 색상을
동기화하여 사용자가 어떤 Hopping이 어떤 행렬 원소에
기여하는지 직관적으로 파악할 수 있게 합니다.

사용 방식:
    color_mgr = ColorManager()
    color = color_mgr.assign_color(hop.uid)
    colored_latex = color_mgr.get_latex_colored(hop.uid, "t \\cdot e^{ikx}")
"""

from __future__ import annotations

from typing import Optional

# ══════════════════════════════════════════════════════════════════════
# 12색 팔레트 — 다크테마 호환, 충분한 구분성
# ══════════════════════════════════════════════════════════════════════

HOPPING_COLOR_PALETTE: list[str] = [
    "#06b6d4",  # cyan
    "#a855f7",  # purple
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#f43f5e",  # rose
    "#3b82f6",  # blue
    "#84cc16",  # lime
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#8b5cf6",  # violet
    "#22d3ee",  # sky
]

# On-site 전용 색상 (항상 고정)
ONSITE_COLOR: str = "#f43f5e"


class ColorManager:
    """
    Hopping UID → 색상 매핑 관리.

    다이어그램의 Edge 색상과 LaTeX 수식의 색상을 일관되게 유지합니다.
    매 렌더링 사이클마다 새로 생성하거나, Session State에 저장하여 유지할 수 있습니다.
    """

    def __init__(self) -> None:
        # uid → color hex 매핑
        self._uid_to_color: dict[str, str] = {}
        # 다음에 할당할 팔레트 인덱스
        self._next_idx: int = 0

    def assign_color(self, hop_uid: str, is_onsite: bool = False) -> str:
        """
        Hopping UID에 색상을 할당하고 반환.

        이미 할당된 경우 기존 색상을 반환합니다.
        On-site hopping은 항상 고정 색상(ONSITE_COLOR)을 사용합니다.

        Args:
            hop_uid: Hopping의 고유 식별자
            is_onsite: On-site hopping 여부

        Returns:
            Hex 색상 문자열 (예: "#06b6d4")
        """
        if is_onsite:
            self._uid_to_color[hop_uid] = ONSITE_COLOR
            return ONSITE_COLOR

        if hop_uid in self._uid_to_color:
            return self._uid_to_color[hop_uid]

        color = HOPPING_COLOR_PALETTE[self._next_idx % len(HOPPING_COLOR_PALETTE)]
        self._uid_to_color[hop_uid] = color
        self._next_idx += 1
        return color

    def get_color(self, hop_uid: str) -> str:
        """이미 할당된 색상을 조회. 없으면 기본 slate 색상 반환."""
        return self._uid_to_color.get(hop_uid, "#94a3b8")

    def get_latex_colored(self, hop_uid: str, latex_str: str) -> str:
        """
        LaTeX 수식을 해당 Hopping의 색상으로 래핑.

        Streamlit의 st.markdown에서 HTML 색상을 사용합니다.
        (LaTeX \\color 명령은 Streamlit KaTeX에서 hex 색상 지원이 제한적이므로
         HTML <span> 래핑을 병행합니다.)

        Args:
            hop_uid: Hopping UID
            latex_str: 원본 LaTeX 문자열

        Returns:
            색상이 적용된 LaTeX 문자열
        """
        color = self.get_color(hop_uid)
        # KaTeX에서 사용할 수 있는 \\textcolor{hex}{...} 형식
        return rf"\textcolor{{{color}}}{{{latex_str}}}"

    def get_html_colored(self, hop_uid: str, text: str) -> str:
        """
        일반 텍스트를 해당 Hopping의 색상으로 HTML 래핑.

        Args:
            hop_uid: Hopping UID
            text: 표시할 텍스트

        Returns:
            <span style="color:...">text</span> 형태의 HTML
        """
        color = self.get_color(hop_uid)
        return f'<span style="color:{color}; font-weight:600;">{text}</span>'

    def build_legend_html(self, hop_entries: list[tuple[str, str, str]]) -> str:
        """
        색상 범례를 HTML로 생성.

        Args:
            hop_entries: [(uid, label, amplitude_str), ...] 리스트

        Returns:
            HTML 문자열
        """
        items = []
        for uid, label, amp in hop_entries:
            color = self.get_color(uid)
            items.append(
                f'<div style="display:flex; align-items:center; gap:8px; margin:2px 0;">'
                f'<div style="width:14px; height:14px; border-radius:3px; '
                f'background:{color}; flex-shrink:0;"></div>'
                f'<span style="color:#e2e8f0; font-size:0.82rem;">'
                f'{label} → t={amp}</span></div>'
            )
        return "\n".join(items)

    def reset(self) -> None:
        """모든 색상 할당을 초기화."""
        self._uid_to_color.clear()
        self._next_idx = 0
