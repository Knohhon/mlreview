"""Каталог метрик: применимость к задаче, направление, обоснование и ловушки (FR-M02).

Только данные — без вычислений, поэтому каталог проверяет раздел `metrics:` спецификации
задачи без тяжёлых импортов. Поведение зафиксировано в `docs/specs/metrics.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

BINARY = "classification/binary"
MULTI = "classification/multiclass"
REGRESSION = "regression"


@dataclass(frozen=True)
class Param:
    """Параметр метрики `имя@значение`."""

    name: str
    default: float
    integer: bool
    low: float
    high: float | None
    inclusive_low: bool

    def check(self, value: float) -> str | None:
        """Сообщение об ошибке или `None`."""
        if self.integer and value != int(value):
            return f"{self.name} — целое число"
        too_low = value < self.low if self.inclusive_low else value <= self.low
        too_high = self.high is not None and value >= self.high
        if too_low or too_high:
            if self.high is None:
                return f"{self.name} ≥ {self.low:g}"
            return f"{self.name} в ({self.low:g}, {self.high:g})"
        return None


K = Param("k", 10, True, 1, None, True)
BETA = Param("β", 1.0, False, 0, None, False)
TAU = Param("τ", 0.5, False, 0, 1, False)


@dataclass(frozen=True)
class MetricInfo:
    """Описание метрики каталога."""

    name: str
    title: str
    tasks: frozenset[tuple[str, str]]
    kind: str  # classes | scores | probabilities | values
    higher_is_better: bool
    bounded: str | None  # high — лучшее 1 на [0, 1]; low — лучшее 0 на [0, 1]
    rationale: str
    pitfalls: tuple[str, ...]
    param: Param | None = None


def _tasks(*pairs: tuple[str, str]) -> frozenset[tuple[str, str]]:
    return frozenset(pairs)


_CLASS_BINARY = _tasks((BINARY, "labels"), (BINARY, "probabilities"))
_CLASS_ALL = _CLASS_BINARY | _tasks((MULTI, "labels"), (MULTI, "probabilities"), (MULTI, "scores"))
_SCORE_BINARY = _tasks((BINARY, "probabilities"), (BINARY, "scores"))
_PROB = _tasks((BINARY, "probabilities"), (MULTI, "probabilities"))
_REG = _tasks((REGRESSION, "values"))

CATALOG: dict[str, MetricInfo] = {
    info.name: info
    for info in (
        MetricInfo(
            "accuracy",
            "доля верных предсказаний",
            _CLASS_ALL,
            "classes",
            True,
            "high",
            "классы сбалансированы и ошибки разных видов стоят одинаково",
            (
                "при дисбалансе константный предиктор получает долю мажоритарного класса (B05)",
                "не учитывает цену ошибок",
            ),
        ),
        MetricInfo(
            "balanced_accuracy",
            "средняя полнота по классам",
            _CLASS_ALL,
            "classes",
            True,
            "high",
            "дисбаланс классов, а ошибки на редком классе важны так же, как на частом",
            ("не учитывает точность: много ложных тревог почти не штрафуется",),
        ),
        MetricInfo(
            "precision",
            "точность класса 1",
            _CLASS_BINARY,
            "classes",
            True,
            "high",
            "ложная тревога дорога: важно, чтобы срабатывания были верными",
            ("не определена, если модель не предсказала ни одного класса 1",),
        ),
        MetricInfo(
            "recall",
            "полнота класса 1",
            _CLASS_BINARY,
            "classes",
            True,
            "high",
            "пропуск дорог: важно найти как можно больше примеров класса 1",
            ("«всегда класс 1» получает 1.0 — смотрите вместе с точностью",),
        ),
        MetricInfo(
            "f1",
            "гармоническое среднее точности и полноты",
            _CLASS_BINARY,
            "classes",
            True,
            "high",
            "пропуск и ложная тревога стоят примерно одинаково, класс 1 редкий",
            ("зависит от порога", "не учитывает верные отрицательные"),
        ),
        MetricInfo(
            "fbeta",
            "F-мера с весом полноты β",
            _CLASS_BINARY,
            "classes",
            True,
            "high",
            "пропуск дороже ложной тревоги в r раз: β = √r",
            ("зависит от порога", "β задаётся из цены ошибок — пересматривайте при её изменении"),
            BETA,
        ),
        MetricInfo(
            "macro_f1",
            "среднее F1 по классам",
            _tasks((MULTI, "labels"), (MULTI, "probabilities"), (MULTI, "scores")),
            "classes",
            True,
            "high",
            "все классы важны одинаково, включая редкие",
            ("шумная на редких классах: смотрите интервалы",),
        ),
        MetricInfo(
            "roc_auc",
            "площадь под ROC-кривой",
            _SCORE_BINARY,
            "scores",
            True,
            "high",
            "важно качество ранжирования без выбора порога",
            (
                "при сильном дисбалансе оптимистична — смотрите pr_auc",
                "не чувствительна к калибровке",
            ),
        ),
        MetricInfo(
            "pr_auc",
            "average precision (площадь под PR-кривой)",
            _SCORE_BINARY,
            "scores",
            True,
            "high",
            "класс 1 редкий, а важны верхние позиции ранжирования",
            ("зависит от доли класса 1 — сравнивайте только на одной выборке",),
        ),
        MetricInfo(
            "precision_at_k",
            "доля класса 1 среди K лучших по оценке",
            _SCORE_BINARY,
            "scores",
            True,
            "high",
            "человек проверяет K первых объектов очереди",
            ("при K больше числа примеров класса 1 не может быть 1.0",),
            K,
        ),
        MetricInfo(
            "recall_at_k",
            "доля класса 1, попавшая в K лучших",
            _SCORE_BINARY,
            "scores",
            True,
            "high",
            "важно, какую часть класса 1 покрывает бюджет проверки K",
            ("при малом K ограничена сверху K / число примеров класса 1",),
            K,
        ),
        MetricInfo(
            "top_k_accuracy",
            "верный класс среди K самых вероятных",
            _tasks((MULTI, "probabilities"), (MULTI, "scores")),
            "scores",
            True,
            "high",
            "пользователь выбирает из K подсказок",
            ("при K, близком к числу классов, вырождается в 1.0",),
            K,
        ),
        MetricInfo(
            "log_loss",
            "логарифмическая функция потерь",
            _PROB,
            "probabilities",
            False,
            None,
            "вероятности используются напрямую: важны и разделимость, и калибровка",
            ("сильно штрафует уверенные ошибки; чувствительна к шумным меткам",),
        ),
        MetricInfo(
            "brier",
            "средний квадрат ошибки вероятности",
            _tasks((BINARY, "probabilities")),
            "probabilities",
            False,
            "low",
            "важна точность вероятностей, а не только ранжирование",
            ("при дисбалансе мала даже у константы",),
        ),
        MetricInfo(
            "ece",
            "ожидаемая ошибка калибровки (10 корзин)",
            _PROB,
            "probabilities",
            False,
            "low",
            "вероятность показывают человеку или подставляют в расчёт",
            (
                "константа доли класса 1 идеально откалибрована — не отражает разделимость",
                "зависит от числа корзин",
            ),
        ),
        MetricInfo(
            "mae",
            "средняя абсолютная ошибка",
            _REG,
            "values",
            False,
            None,
            "ошибки стоят пропорционально размеру, выбросы не должны доминировать",
            ("не различает недооценку и переоценку",),
        ),
        MetricInfo(
            "rmse",
            "корень из среднего квадрата ошибки",
            _REG,
            "values",
            False,
            None,
            "крупные ошибки непропорционально дороже",
            ("доминируется выбросами",),
        ),
        MetricInfo(
            "median_ae",
            "медиана абсолютной ошибки",
            _REG,
            "values",
            False,
            None,
            "важна типичная ошибка, выбросы не важны",
            ("не видит редкие крупные ошибки",),
        ),
        MetricInfo(
            "mape",
            "средняя абсолютная процентная ошибка",
            _REG,
            "values",
            False,
            None,
            "важна относительная ошибка, метки далеки от нуля",
            ("не определена при нулевых метках", "сильнее штрафует переоценку малых значений"),
        ),
        MetricInfo(
            "pinball",
            "квантильная функция потерь уровня τ",
            _REG,
            "values",
            False,
            None,
            "недооценка дороже переоценки в r раз: τ = r / (1 + r)",
            ("оценивает квантиль, а не среднее — сравнивайте модели с одним τ",),
            TAU,
        ),
        MetricInfo(
            "r2",
            "коэффициент детерминации",
            _REG,
            "values",
            True,
            None,
            "нужна доля объяснённой дисперсии для отчёта",
            ("зависит от дисперсии меток выборки — не сравнивайте между датасетами",),
        ),
    )
}


def parse_name(name: str) -> tuple[str, str | None]:
    """`precision_at_k@50` → (`precision_at_k`, `50`)."""
    base, _, param = name.partition("@")
    return base, (param if param else None)


def param_value(info: MetricInfo, raw: str | None) -> float | None:
    """Значение параметра метрики (по умолчанию — из каталога)."""
    if info.param is None:
        return None
    if raw is None:
        return info.param.default
    return float(raw)


def _param_problem(info: MetricInfo, raw: str | None) -> str | None:
    if raw is None:
        return None
    if info.param is None:
        return f"у метрики {info.name!r} нет параметра"
    try:
        value = float(raw)
    except ValueError:
        return f"параметр {info.param.name} метрики {info.name!r} — число, а не {raw!r}"
    problem = info.param.check(value)
    return f"параметр метрики {info.name!r}: {problem}" if problem else None


def check_name(
    name: str, task_type: str, output: str, custom: frozenset[str] = frozenset()
) -> str | None:
    """Ошибка имени метрики для задачи или `None` (docs/specs/metrics.md, E1)."""
    if name in custom:
        return None
    base, raw = parse_name(name)
    info = CATALOG.get(base)
    allowed = ", ".join(applicable(task_type, output))
    if info is None:
        return f"неизвестная метрика {name!r}; допустимо: {allowed}"
    if (task_type, output) not in info.tasks:
        return f"метрика {base!r} не подходит к {task_type} · {output}; допустимо: {allowed}"
    return _param_problem(info, raw)


def applicable(task_type: str, output: str) -> list[str]:
    """Метрики каталога для пары `тип · выход` (с параметром — `имя@<параметр>`)."""
    return [
        f"{info.name}@{info.param.name}" if info.param else info.name
        for info in CATALOG.values()
        if (task_type, output) in info.tasks
    ]
