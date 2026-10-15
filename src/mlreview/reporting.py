"""Представление отчёта ревью: фильтр важности и лимит, текст, JSON (FR-R01, FR-R05).

Одно представление для CLI, Python API, MCP и проактивного режима; формат зафиксирован
в `docs/specs/review.md` и `docs/specs/findings-store.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlreview.core.findings import Finding, Report, severity_rank
from mlreview.core.store import STATUS_NAMES, Entry, JournalEntry, ReviewRecord, SearchResult
from mlreview.core.workspace import INIT_HINT

NOT_CONNECTED = f"не подключено — {INIT_HINT.split(' — ')[-1]}"
NOT_USED = "не используется (--no-store)"
DETERMINISTIC = "LLM: не используется — отчёт детерминированный"


@dataclass
class ReportView:
    """Что показать из отчёта: находки после фильтров и сведения о хранилище."""

    items: list[tuple[Finding, Entry | None]]
    hidden_dismissed: list[str] = field(default_factory=list)
    hidden_by_filter: int = 0
    store: str | None = None


def build_view(
    report: Report,
    record: ReviewRecord | None = None,
    *,
    min_severity: str = "info",
    limit: int = 0,
    store: str | None = None,
) -> ReportView:
    """Находки для показа: по важности, не ниже порога, не больше лимита (B9)."""
    if record is not None:
        items: list[tuple[Finding, Entry | None]] = [(e.finding, e) for e in record.entries]
        hidden = [entry.id for entry in record.hidden]
    else:
        items = [(finding, None) for finding in report.findings]
        hidden = []
    shown = [
        item for item in items if severity_rank(item[0].severity) <= severity_rank(min_severity)
    ]
    if limit > 0:
        shown = shown[:limit]
    return ReportView(shown, hidden, len(items) - len(shown), store)


def _status_text(entry: Entry) -> str:
    if entry.status == "new" and entry.occurrence > 1:
        return f"повторяется: {entry.occurrence}-е вхождение"
    return STATUS_NAMES.get(entry.status, entry.status)


def _version(version: str | None, data_hash: str | None) -> str:
    text = version or "нет"
    return f"{text} ({data_hash})" if data_hash else text


def render_report(
    report: Report,
    title: str,
    view: ReportView | None = None,
    *,
    llm_line: str = DETERMINISTIC,
) -> str:
    """Текстовый отчёт: заголовок, находки по важности, что скрыто и не проверено, строка LLM."""
    view = view or build_view(report)
    lines = [f"{title} {report.run_id}"]
    if report.baseline_id is not None:
        lines.append(
            f"  базовый прогон: {report.baseline_id}, общих примеров: {report.compared_examples}"
        )
        before = _version(report.baseline_data_version, report.baseline_data_hash)
        version = f"{before} → {_version(report.data_version, report.data_hash)}"
    else:
        version = _version(report.data_version, report.data_hash)
    lines.append(f"  версия данных: {version}")
    if view.store is not None:
        lines.append(f"  хранилище находок: {view.store}")
    lines.append(f"находки: {len(view.items)}" if view.items else "находок нет")
    for number, (finding, entry) in enumerate(view.items, start=1):
        label = f"{entry.id} " if entry is not None else ""
        lines.append(
            f"  {number}. [{finding.severity}] {label}{finding.check} — {finding.message}"
            f" ({finding.requirement})"
        )
        if entry is not None:
            lines.append(f"     статус: {_status_text(entry)}")
            if entry.note:
                lines.append(f"     пометка: {entry.note}")
        if finding.evidence:
            lines.append(f"     примеры: {', '.join(finding.evidence)}")
        if finding.suggestion:
            lines.append(f"     предложение: {finding.suggestion}")
    if view.hidden_dismissed:
        lines.append(f"скрыто отклонённых: {len(view.hidden_dismissed)}")
    if view.hidden_by_filter:
        lines.append(f"скрыто: {view.hidden_by_filter} (ниже порога или сверх лимита)")
    if report.skipped:
        lines.append(f"не проверено: {len(report.skipped)}")
        lines += [f"  - {skipped.check}: {skipped.reason}" for skipped in report.skipped]
    lines.append(llm_line)
    return "\n".join(lines)


def report_payload(report: Report, view: ReportView) -> dict[str, Any]:
    """Отчёт для `--format json`: находки после фильтров с полями хранилища (B8)."""
    payload = report.to_dict()
    payload["findings"] = [
        finding.to_dict() if entry is None else entry.to_dict() for finding, entry in view.items
    ]
    payload["hidden_dismissed"] = view.hidden_dismissed
    payload["hidden_by_filter"] = view.hidden_by_filter
    payload["store"] = view.store if view.store not in (NOT_CONNECTED, NOT_USED) else None
    return payload


KIND_NAMES = {"hypothesis": "гипотеза", "decision": "решение", "note": "заметка"}


def entry_summary(entry: JournalEntry) -> str:
    """Запись журнала одной строкой: id, вид, статус или исход, текст."""
    state = entry.status or entry.outcome
    state_text = f" [{state}]" if state else ""
    return f"{entry.id} {KIND_NAMES[entry.kind]}{state_text} — {entry.text}"


def render_search(result: SearchResult) -> list[str]:
    """Выдача поиска (B3–B7)."""
    if result.hits:
        lines = [f"найдено: {len(result.hits)}"]
        for hit in result.hits:
            lines.append(f"  {entry_summary(hit.entry)}")
            if hit.entry.runs:
                lines.append(f"     прогоны: {', '.join(hit.entry.runs)}")
            if hit.entry.data_versions:
                lines.append(f"     версии данных: {', '.join(hit.entry.data_versions)}")
            lines += [f"     связано: {entry_summary(linked)}" for linked in hit.linked]
            if hit.stale:
                lines.append(f"     {hit.stale}")
        return lines
    if result.config_matches:
        found = ", ".join(f"{run} ({key}={value})" for run, key, value in result.config_matches)
        return [
            f"в журнале записей нет; найдено в конфигах прогонов: {found};"
            " причина решения не зафиксирована"
        ]
    return ["ничего не найдено"]
