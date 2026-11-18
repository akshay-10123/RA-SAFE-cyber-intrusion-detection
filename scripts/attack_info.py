"""Operator-facing attack descriptions and recommended actions (phase 3 XAI alerts)."""
from __future__ import annotations

from dataclasses import dataclass

from scripts.attack_labels import (
    ATTACK_DISPLAY_NAMES,
    CLASS_BENIGN,
    CLASS_DOS,
    CLASS_EVIL_TWIN,
    CLASS_FDI,
    CLASS_REPLAY,
)

CONFIDENCE_ALERT_THRESHOLD = 0.55


@dataclass(frozen=True)
class AttackInfo:
    class_id: int
    key: str
    title: str
    layer: str
    severity: str
    severity_rank: int
    summary: str
    description: str
    indicators: tuple[str, ...]
    recommended_actions: tuple[str, ...]


_ATTACKS: tuple[AttackInfo, ...] = (
    AttackInfo(
        CLASS_BENIGN,
        "benign",
        ATTACK_DISPLAY_NAMES[CLASS_BENIGN],
        "—",
        "None",
        0,
        "Normal flight — no intrusion indicators.",
        "Network and telemetry patterns match expected benign UAV operation. "
        "Continue routine monitoring.",
        ("Stable Wi‑Fi packet timing", "Telemetry within expected flight envelope"),
        ("Continue normal flight operations.", "Maintain periodic IDS checks."),
    ),
    AttackInfo(
        CLASS_DOS,
        "dos",
        ATTACK_DISPLAY_NAMES[CLASS_DOS],
        "Cyber (Wi‑Fi)",
        "Critical",
        4,
        "De-authentication flood disrupting controller–drone link.",
        "An attacker sends de-authentication frames to force the UAV off the wireless network, "
        "causing loss of control link or failsafe behavior.",
        (
            "Sudden disconnect or link flapping",
            "Abnormal wlan.duration / frame patterns",
            "Loss of command channel while motors still active",
        ),
        (
            "Verify access point and enable 802.11w (PMF) if available.",
            "Switch to backup control link or pre-programmed failsafe (RTL/hover).",
            "Physically recover the aircraft if link cannot be restored.",
        ),
    ),
    AttackInfo(
        CLASS_REPLAY,
        "replay",
        ATTACK_DISPLAY_NAMES[CLASS_REPLAY],
        "Cyber (Wi‑Fi)",
        "High",
        3,
        "Previously captured commands re-transmitted to the flight controller.",
        "Replay attacks reuse valid packets from an earlier session. The drone may execute "
        "stale commands while appearing to receive legitimate traffic.",
        (
            "Duplicate or out-of-order wlan.seq values",
            "Command timing inconsistent with current flight phase",
            "Network activity without matching pilot input",
        ),
        (
            "Enable sequence numbers / timestamp validation on commands.",
            "Use encrypted, session-bound control channels.",
            "Consider RTL until the control path is verified.",
        ),
    ),
    AttackInfo(
        CLASS_EVIL_TWIN,
        "evil_twin",
        ATTACK_DISPLAY_NAMES[CLASS_EVIL_TWIN],
        "Cyber (Wi‑Fi)",
        "High",
        3,
        "Rogue access point impersonating the legitimate controller AP.",
        "The attacker presents a fake Wi‑Fi endpoint with a similar SSID/BSSID footprint. "
        "The UAV or operator may associate with the malicious node.",
        (
            "Unexpected BSSID or signal anomalies",
            "MITM-style routing of traffic",
            "Association to unknown AP while mission active",
        ),
        (
            "Whitelist authorized AP MAC addresses.",
            "Disconnect and re-pair using out-of-band verification.",
            "Abort mission if association identity cannot be confirmed.",
        ),
    ),
    AttackInfo(
        CLASS_FDI,
        "fdi",
        ATTACK_DISPLAY_NAMES[CLASS_FDI],
        "Physical / cyber-physical",
        "Critical",
        4,
        "False telemetry or control values injected into the flight stack.",
        "False Data Injection corrupts sensor or control inputs — pitch, roll, yaw, or velocity "
        "may deviate without a matching cyber signature alone.",
        (
            "Telemetry spikes inconsistent with flight dynamics",
            "Sensor fusion disagreement (IMU vs expected motion)",
            "Control response without valid command path",
        ),
        (
            "Engage sensor plausibility checks and bounded control rates.",
            "Switch to redundant sensors or conservative flight mode.",
            "Land immediately if attitude/altitude diverges from pilot intent.",
        ),
    ),
)

BY_ID: dict[int, AttackInfo] = {a.class_id: a for a in _ATTACKS}
BY_KEY: dict[str, AttackInfo] = {a.key: a for a in _ATTACKS}


def get_attack_info(class_id: int) -> AttackInfo:
    if class_id not in BY_ID:
        raise ValueError(f"Unknown class_id: {class_id}")
    return BY_ID[class_id]


def all_attack_infos() -> list[AttackInfo]:
    return list(_ATTACKS)


def is_intrusion(class_id: int, confidence: float) -> bool:
    if class_id == CLASS_BENIGN:
        return False
    return confidence >= CONFIDENCE_ALERT_THRESHOLD


def alert_headline(class_id: int, confidence: float) -> str:
    if class_id == CLASS_BENIGN:
        return "Status: Normal operation"
    if confidence >= CONFIDENCE_ALERT_THRESHOLD:
        return "INTRUSION DETECTED"
    return "Possible anomaly — low confidence"
