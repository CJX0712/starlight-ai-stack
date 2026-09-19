"""配置与模型档位。

作者: 晨星

档位存在的意义：同一套代码在「内存充裕」和「内存紧张」的机器上都能跑，
切换只改一个环境变量 STARLIGHT_PROFILE，不改任何业务代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOME = PROJECT_ROOT / "data"

AUTHOR = "晨星"


@dataclass(frozen=True)
class ModelProfile:
    """一组锁定的模型组合。llm / embed 都是 Ollama 中的模型名。"""

    name: str
    llm: str
    embed: str
    ctx: int
    num_thread: int
    max_tokens: int
    temperature: float
    note: str = ""


PROFILES: dict[str, ModelProfile] = {
    "high": ModelProfile(
        name="high",
        llm="qwen2.5:7b-instruct-q4_K_M",
        embed="bge-m3",
        ctx=4096,
        num_thread=4,
        max_tokens=640,
        temperature=0.2,
        note="最高质量档，需要约 5GB 以上空闲内存，CPU 上约 2-3 tok/s",
    ),
    "balanced": ModelProfile(
        name="balanced",
        llm="qwen3:4b",
        embed="nomic-embed-text",
        ctx=4096,
        num_thread=4,
        max_tokens=512,
        temperature=0.2,
        note="默认档：本机已就绪的组合，实测 4-5 tok/s；bge-m3 拉取完成后可把 embed 换成 bge-m3",
    ),
    "lite": ModelProfile(
        name="lite",
        llm="qwen3:4b",
        embed="nomic-embed-text",
        ctx=2048,
        num_thread=2,
        max_tokens=256,
        temperature=0.2,
        note="内存紧张时的降级档，缩短上下文与生成长度",
    ),
}

# 默认尝试最高档：模型齐备且内存够就用 7B，否则 resolve_profile 会自动降级
DEFAULT_PROFILE = "high"

# 模型缺失时的降级顺序：本机 Ollama 里有什么就用什么，链路优先于纸面配置
LLM_FALLBACKS = ["qwen2.5:7b-instruct-q4_K_M", "qwen3:4b"]
EMBED_FALLBACKS = ["bge-m3", "nomic-embed-text"]


def free_memory_gb() -> float:
    """读取物理可用内存（GB）。读不到返回 -1，表示不做内存守卫。"""
    try:
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / (1024 ** 3)
    except Exception:
        return -1.0


# 高配档所需的最小可用内存：7B Q4 权重约 4.7GB，低于此值会疯狂换页
HIGH_MIN_FREE_GB = 4.0


def resolve_profile(
    models_info: dict[str, list[str]], profile: ModelProfile
) -> tuple[ModelProfile, list[str]]:
    """按本机实际可用模型修正档位，返回（生效档位，调整说明）。

    存在意义：配置里写的模型可能没拉下来（网络慢、磁盘不够、换机器），
    与其让整条链路崩掉，不如自动降级到可用模型并把原因讲清楚。
    """
    notes: list[str] = []
    available = set(models_info)

    def pick(want: str, fallbacks: list[str], capability: str) -> str:
        if want in available:
            return want
        for cand in fallbacks:
            if cand in available:
                notes.append(f"{want} 不可用，降级为 {cand}")
                return cand
        for name, caps in models_info.items():
            if capability in caps:
                notes.append(f"{want} 与预设备选均不可用，改用本机已有的 {name}")
                return name
        notes.append(f"本机没有可用的 {capability} 模型")
        return want

    llm = pick(profile.llm, LLM_FALLBACKS, "completion")
    embed = pick(profile.embed, EMBED_FALLBACKS, "embedding")

    # 内存守卫：模型存在但内存不够时同样要降级，否则整机换页卡死
    free = free_memory_gb()
    if profile.name == "high" and 0 < free < HIGH_MIN_FREE_GB:
        notes.append(f"可用内存 {free:.1f}GB 低于高配档下限 {HIGH_MIN_FREE_GB}GB，降级为 balanced")
        profile = PROFILES["balanced"]
        llm = pick(profile.llm, LLM_FALLBACKS, "completion")
        embed = pick(profile.embed, EMBED_FALLBACKS, "embedding")

    if llm == profile.llm and embed == profile.embed:
        return profile, notes
    return (
        ModelProfile(
            name=profile.name, llm=llm, embed=embed, ctx=profile.ctx,
            num_thread=profile.num_thread, max_tokens=profile.max_tokens,
            temperature=profile.temperature, note=profile.note,
        ),
        notes,
    )


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """全局设置。全部可被环境变量覆盖，便于容器与 CI 复现。"""

    profile: str = field(default_factory=lambda: os.getenv("STARLIGHT_PROFILE", DEFAULT_PROFILE))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("STARLIGHT_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    home: Path = field(
        default_factory=lambda: Path(os.getenv("STARLIGHT_HOME", str(DEFAULT_HOME)))
    )
    chunk_size: int = field(default_factory=lambda: _env_int("STARLIGHT_CHUNK_SIZE", 420))
    chunk_overlap: int = field(default_factory=lambda: _env_int("STARLIGHT_CHUNK_OVERLAP", 80))
    top_k: int = field(default_factory=lambda: _env_int("STARLIGHT_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("STARLIGHT_CANDIDATE_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("STARLIGHT_RRF_K", 60))
    timeout_s: float = field(default_factory=lambda: _env_float("STARLIGHT_TIMEOUT", "300"))
    api_host: str = field(default_factory=lambda: os.getenv("STARLIGHT_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("STARLIGHT_API_PORT", 8787))

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(
                f"未知档位 {self.profile!r}，可选: {sorted(PROFILES)}"
            )
        self.home = Path(self.home)
        self.home.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> ModelProfile:
        return PROFILES[self.profile]

    @property
    def db_path(self) -> Path:
        return self.home / "starlight.db"

    def describe(self) -> dict:
        p = self.model
        return {
            "profile": p.name,
            "llm": p.llm,
            "embed": p.embed,
            "ctx": p.ctx,
            "num_thread": p.num_thread,
            "note": p.note,
            "home": str(self.home),
            "db": str(self.db_path),
            "ollama": self.ollama_base_url,
        }
