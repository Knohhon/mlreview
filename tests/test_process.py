"""Логические проверки процесса (FR-D06, B07, B11).

Спецификация: docs/specs/process-checks.md.
"""

from __future__ import annotations

from _factories import binary_contract, binary_spec
from mlreview.analyzers.process import process_findings
from mlreview.analyzers.registry import ReviewContext


def _run(*args, **kwargs):
    ctx_kwargs = {
        key: kwargs.pop(key)
        for key in ("candidate_seeds", "baseline_seeds", "prior_runs")
        if key in kwargs
    }
    return process_findings(ReviewContext(binary_spec(), *args, **ctx_kwargs))


def _by_check(results):
    return {finding.check: finding for finding in results.findings}


def test_different_data_versions():
    """B1 (B07): разные версии и хеши данных."""
    candidate = binary_contract("cand", data_version="v2", shift=0.5)
    baseline = binary_contract("base", seed=2, data_version="v1")
    finding = _by_check(_run(candidate, baseline))["process.data_version_mismatch"]
    assert finding.severity == "high"
    assert finding.message == (
        "сравнение на разных версиях данных: v1 → v2 — разница метрик может объясняться"
        " данными, а не моделью"
    )
    assert finding.facts == {
        "baseline_version": "v1",
        "candidate_version": "v2",
        "baseline_hash": baseline.data_hash,
        "candidate_hash": candidate.data_hash,
    }
    assert "различающий эксперимент" in finding.suggestion
    assert finding.requirement == "FR-D06, B07"


def test_same_version_different_content():
    """B2: версия одна, хеши разные; версия не задана."""
    candidate = binary_contract("cand", data_version="v1", shift=0.5)
    baseline = binary_contract("base", data_version="v1")
    finding = _by_check(_run(candidate, baseline))["process.data_version_mismatch"]
    assert finding.message == (
        f"сравнение на разных данных: хеш {baseline.data_hash} → {candidate.data_hash}"
        " при версии v1 — метка версии не отражает изменение"
    )
    unversioned = _run(
        binary_contract("cand", data_version=None, shift=0.5),
        binary_contract("base", data_version=None),
    )
    message = _by_check(unversioned)["process.data_version_mismatch"].message
    assert "при версии не задана" in message


def test_different_labels_same_content_is_fine():
    """B3: версии разные, данные одинаковые — находки нет."""
    results = _run(binary_contract("cand", data_version="v2"), binary_contract("base", seed=2))
    assert "process.data_version_mismatch" not in _by_check(results)


def test_multiple_changes():
    """B4 (B11): три параметра сразу; сид не считается."""
    candidate = binary_contract(
        "cand", config={"loss": "focal", "lr": 0.001, "augment": "flip", "seed": 2}
    )
    baseline = binary_contract(
        "base", config={"loss": "ce", "lr": 0.01, "augment": "none", "seed": 1}
    )
    finding = _by_check(_run(candidate, baseline))["process.multiple_changes"]
    assert finding.severity == "medium"
    assert finding.message == (
        "в эксперименте изменено сразу 3 параметра: augment (none → flip), loss (ce → focal),"
        " lr (0.01 → 0.001) — эффект нельзя приписать одному изменению"
    )
    assert finding.facts == {"changed": 3, "keys": "augment, loss, lr"}
    assert finding.requirement == "FR-D06, B11"


def test_one_change_or_seed_only_is_fine():
    """B5: один параметр или только сид — находки нет."""
    one = _run(
        binary_contract("cand", config={"lr": 0.1, "seed": 1}),
        binary_contract("base", config={"lr": 0.2, "seed": 2}),
    )
    assert "process.multiple_changes" not in _by_check(one)
    nested_seed = _run(
        binary_contract("cand", config={"model": {"random_state": 1}, "lr": 0.1}),
        binary_contract("base", config={"model": {"random_state": 2}, "lr": 0.1}),
    )
    assert "process.multiple_changes" not in _by_check(nested_seed)


def test_key_on_one_side_and_nested_keys():
    """B6: ключ только у одного прогона; вложенные ключи через точку; строки и числа."""
    candidate = binary_contract("cand", config={"optimizer": {"lr": 0.01}, "dropout": 0.1})
    baseline = binary_contract("base", config={"optimizer": {"lr": "0.02"}})
    finding = _by_check(_run(candidate, baseline))["process.multiple_changes"]
    assert "dropout (нет → 0.1), optimizer.lr (0.02 → 0.01)" in finding.message
    same = _run(
        binary_contract("cand", config={"lr": 0.01, "loss": "ce"}),
        binary_contract("base", config={"lr": "0.01", "loss": "ce"}),
    )
    assert "process.multiple_changes" not in _by_check(same)


def test_seed_config_mismatch():
    """B7: сиды кандидата различаются ещё и lr."""
    candidate = binary_contract("c1", config={"lr": 0.1, "seed": 1})
    seeds = (
        binary_contract("c2", config={"lr": 0.1, "seed": 2}),
        binary_contract("c3", config={"lr": 0.2, "seed": 3}),
    )
    finding = _by_check(_run(candidate, None, candidate_seeds=seeds))[
        "process.seed_config_mismatch"
    ]
    assert finding.severity == "medium"
    assert finding.message == "прогоны по сидам кандидата различаются не только сидом: lr"


def test_validation_reuse():
    """B8: 20 прогонов на одних данных — находка; 19 — нет."""
    contract = binary_contract()
    finding = _by_check(_run(contract, None, prior_runs=19))["process.validation_reuse"]
    assert finding.severity == "medium"
    assert finding.message.startswith(
        f"на одних данных ({contract.data_hash}) прошли ревью 20 прогонов"
    )
    assert finding.facts == {"runs": 20, "data_hash": contract.data_hash, "threshold": 20}
    assert "process.validation_reuse" not in _by_check(_run(contract, None, prior_runs=18))


def test_skips():
    """E1–E3: нет базового, нет config, нет хранилища."""
    reasons = {s.check: s.reason for s in _run(binary_contract(), None).skipped}
    assert reasons["process.data_version_mismatch"] == "нет базового прогона"
    assert reasons["process.multiple_changes"] == "нет базового прогона"
    assert reasons["process.validation_reuse"] == (
        "нет хранилища находок — история прогонов неизвестна (mlreview init)"
    )
    no_config = _run(binary_contract("cand", config={"lr": 1}), binary_contract("base"))
    reasons = {s.check: s.reason for s in no_config.skipped}
    assert reasons["process.multiple_changes"] == "нет config у базового прогона"
