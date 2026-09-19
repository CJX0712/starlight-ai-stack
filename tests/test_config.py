"""配置与降级决策测试。

作者: 晨星

这些用例锁死的是"降级只在真的需要时才发生"：
- 模型缺失 -> 必须降级
- 内存不足且模型未加载 -> 必须降级
- 模型已常驻内存 -> 绝不因为可用内存低而降级（低可用内存是结果，不是风险）
"""

from starlight import config as cfg


def _info():
    return {
        "qwen2.5:7b-instruct-q4_K_M": ["completion"],
        "qwen3:4b": ["completion", "thinking"],
        "bge-m3": ["embedding"],
    }


def test_missing_model_falls_back(monkeypatch):
    monkeypatch.setattr(cfg, "free_memory_gb", lambda: -1.0)
    info = {"qwen3:4b": ["completion"], "bge-m3": ["embedding"]}
    profile, notes = cfg.resolve_profile(info, cfg.PROFILES["high"], loaded_models=[])
    assert profile.llm == "qwen3:4b"
    assert any("不可用" in n for n in notes)


def test_downgrade_when_memory_low_and_not_loaded(monkeypatch):
    monkeypatch.setattr(cfg, "free_memory_gb", lambda: 1.0)
    profile, notes = cfg.resolve_profile(_info(), cfg.PROFILES["high"], loaded_models=[])
    assert profile.name == "balanced"
    assert any("低于高配档下限" in n for n in notes)


def test_no_downgrade_when_model_already_resident(monkeypatch):
    monkeypatch.setattr(cfg, "free_memory_gb", lambda: 1.0)
    profile, notes = cfg.resolve_profile(
        _info(), cfg.PROFILES["high"], loaded_models=["qwen2.5:7b-instruct-q4_K_M"]
    )
    assert profile.name == "high"
    assert profile.llm == "qwen2.5:7b-instruct-q4_K_M"
    assert any("已常驻内存" in n for n in notes)


def test_profile_keeps_threads_and_context(monkeypatch):
    monkeypatch.setattr(cfg, "free_memory_gb", lambda: 8.0)
    profile, notes = cfg.resolve_profile(_info(), cfg.PROFILES["high"], loaded_models=[])
    assert profile.num_thread == cfg.PROFILES["high"].num_thread
    assert profile.ctx == cfg.PROFILES["high"].ctx
    assert notes == []
