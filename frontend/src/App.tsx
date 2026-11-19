import React, { useEffect, useState } from "react";
import {
    Link,
    NavLink,
    Navigate,
    Route,
    Routes,
    useLocation,
    useNavigate,
    useParams,
    useSearchParams,
} from "react-router-dom";
import { parse, stringify } from "yaml";
import {
    api,
    id,
    setToken,
    Workspace,
    Run,
    Source,
    Finding,
    Entry,
    Task,
    Job,
    Page,
} from "./api";

const nav = [
    ["/overview", "Обзор", "◈"],
    ["/runs", "Прогоны", "▤"],
    ["/findings", "Находки", "◇"],
    ["/journal", "Журнал", "▧"],
    ["/task", "Задача", "⌘"],
    ["/sources", "Источники", "⊞"],
    ["/settings", "Настройки", "⚙"],
];
const names: Record<string, string> = {
    new: "Новая",
    accepted: "Принята",
    dismissed: "Отклонена",
    fixed: "Исправлена",
    queued: "В очереди",
    running: "Выполняется",
    completed: "Завершено",
    failed: "Ошибка",
    interrupted: "Прервано",
    partial: "Неполный",
    valid: "Валиден",
    invalid: "Невалиден",
    unknown: "Неизвестно",
    not_run: "Не запускалось",
    critical: "Критическая",
    high: "Высокая",
    medium: "Средняя",
    low: "Низкая",
    info: "Информация",
    finished: "Завершён",
    killed: "Остановлен",
    scheduled: "Запланирован",
    hypothesis: "Гипотеза",
    decision: "Решение",
    note: "Заметка",
    data: "Данные",
    segments: "Сегменты",
    metrics: "Метрики",
    model: "Модель",
    curves: "Кривые",
    process: "Процесс",
};
const show = (value: unknown) =>
    value === null || value === undefined
        ? "—"
        : typeof value === "object"
          ? JSON.stringify(value)
          : String(value);
function Badge({ value }: { value: string }) {
    return <span className={"badge " + value}>{names[value] || value}</span>;
}
function useData<T>(path: string) {
    const [data, setData] = useState<T>(),
        [error, setError] = useState(""),
        [tick, setTick] = useState(0);
    useEffect(() => {
        let live = true;
        setData(undefined);
        setError("");
        api<T>(path)
            .then((x) => {
                if (live) setData(x);
            })
            .catch((e) => {
                if (live) setError(e.message);
            });
        return () => {
            live = false;
        };
    }, [path, tick]);
    return { data, error, reload: () => setTick((x) => x + 1) };
}
function ErrorBox({ text }: { text: string }) {
    return text ? (
        <div className="error" role="alert">
            {text}
        </div>
    ) : null;
}
function Wait({ error }: { error?: string }) {
    return error ? (
        <ErrorBox text={error} />
    ) : (
        <p className="muted" role="status">
            Загрузка…
        </p>
    );
}
function Heading({
    title,
    sub,
    children,
}: {
    title: string;
    sub?: string;
    children?: React.ReactNode;
}) {
    return (
        <div className="heading">
            <div>
                <p className="eyebrow">
                    РАБОЧЕЕ ПРОСТРАНСТВО / {title.toUpperCase()}
                </p>
                <h1>{title}</h1>
                {sub && <p className="muted">{sub}</p>}
            </div>
            <div className="actions">{children}</div>
        </div>
    );
}
function Json({ value }: { value: unknown }) {
    return <pre>{JSON.stringify(value, null, 2)}</pre>;
}
function Empty({ children }: { children: React.ReactNode }) {
    return (
        <div className="empty">
            <span className="empty-icon">◇</span>
            {children}
        </div>
    );
}
function RunLink({ run }: { run: Run }) {
    return (
        <Link to={"/runs/" + id(run.id)}>
            {run.name}
            <small>{run.native_id}</small>
        </Link>
    );
}
function FindingsTable({ items }: { items: Finding[] }) {
    return items.length ? (
        <div className="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Важность</th>
                        <th>Находка</th>
                        <th>Статус</th>
                        <th>Проверка</th>
                    </tr>
                </thead>
                <tbody>
                    {items.map((f) => (
                        <tr key={f.id}>
                            <td>
                                <Badge value={f.severity} />
                            </td>
                            <td>
                                <Link to={"/findings/" + id(f.id)}>
                                    <small>{f.id}</small>
                                    {f.message}
                                </Link>
                            </td>
                            <td>
                                <Badge value={f.status} />
                            </td>
                            <td className="mono">{f.check}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    ) : (
        <p className="muted">
            Находок в этом списке нет. Это не означает, что все проверки
            выполнены.
        </p>
    );
}
function JobStatus({
    jobId,
    onDone,
}: {
    jobId: string;
    onDone?: (job: Job) => void;
}) {
    const [job, setJob] = useState<Job>(),
        [error, setError] = useState("");
    useEffect(() => {
        let live = true,
            timer: ReturnType<typeof setTimeout>;
        const poll = () =>
            api<Job>("/jobs/" + id(jobId))
                .then((j) => {
                    if (!live) return;
                    setJob(j);
                    if (["queued", "running"].includes(j.status))
                        timer = setTimeout(poll, 700);
                    else onDone?.(j);
                })
                .catch((e) => {
                    if (live) setError(e.message);
                });
        poll();
        return () => {
            live = false;
            clearTimeout(timer);
        };
    }, [jobId]);
    return (
        <div className="job" role="status">
            <span>Операция</span>{" "}
            {job ? <Badge value={job.status} /> : "Загрузка…"}
            <ErrorBox text={error || job?.error || ""} />
        </div>
    );
}
function Pager({
    total,
    offset,
    onChange,
}: {
    total: number;
    offset: number;
    onChange: (n: number) => void;
}) {
    return (
        <div className="pager">
            <span>
                {total ? offset + 1 : 0}–{Math.min(offset + 50, total)} из{" "}
                {total}
            </span>
            <button
                disabled={offset === 0}
                onClick={() => onChange(Math.max(0, offset - 50))}
            >
                ← Назад
            </button>
            <button
                disabled={offset + 50 >= total}
                onClick={() => onChange(offset + 50)}
            >
                Далее →
            </button>
        </div>
    );
}

function Overview() {
    const { data, error } = useData<any>("/overview");
    return (
        <>
            <Heading
                title="Обзор проекта"
                sub="Эксперименты, факты и следующие решения"
            >
                <Link className="button primary" to="/runs">
                    Открыть прогоны →
                </Link>
            </Heading>
            {!data ? (
                <Wait error={error} />
            ) : (
                <>
                    <div className="stats">
                        {[
                            [data.runs, "Прогонов в индексе"],
                            [data.reviewed, "Прогонов с Web-отчётом"],
                            [data.open_critical, "Новых критических"],
                            [data.open_high, "Новых высокой важности"],
                        ].map(([v, k]) => (
                            <Link
                                to={
                                    String(k).includes("важности") ||
                                    String(k).includes("критических")
                                        ? "/findings?status=new"
                                        : "/runs"
                                }
                                className="stat"
                                key={k}
                            >
                                <strong>{v}</strong>
                                <span>{k}</span>
                            </Link>
                        ))}
                    </div>
                    <div className="grid">
                        <section>
                            <h2>Последние прогоны</h2>
                            {data.recent_runs?.length ? (
                                data.recent_runs.map((r: Run) => (
                                    <div className="row" key={r.id}>
                                        <RunLink run={r} />
                                        <Badge value={r.status} />
                                    </div>
                                ))
                            ) : (
                                <Empty>
                                    <h3>Начните с источника данных</h3>
                                    <p>
                                        Подключите каталог прогонов или MLflow.
                                    </p>
                                    <Link to="/sources">
                                        Подключить источник →
                                    </Link>
                                </Empty>
                            )}
                        </section>
                        <section>
                            <h2>Решения</h2>
                            {data.decisions?.length ? (
                                data.decisions.map((e: Entry) => (
                                    <p key={e.id}>
                                        <Link to={"/journal/" + e.id}>
                                            {e.text}
                                        </Link>
                                    </p>
                                ))
                            ) : (
                                <p className="muted">
                                    Решений пока нет. Сохраните вывод после
                                    ревью в журнал.
                                </p>
                            )}
                            <Link to="/journal">Перейти в журнал →</Link>
                        </section>
                    </div>
                    <section>
                        <h2>Требуют внимания</h2>
                        <FindingsTable items={data.findings || []} />
                    </section>
                </>
            )}
        </>
    );
}

function Runs() {
    const [query, setQuery] = useSearchParams(),
        [selected, setSelected] = useState<string[]>([]);
    const { data, error, reload } = useData<Page<Run>>("/runs?" + query),
        sources = useData<Source[]>("/sources"),
        task = useData<Task>("/task");
    const metricNames = Array.from(
        new Set(
            [
                task.data?.parsed?.metrics?.primary,
                ...Object.keys(data?.items?.[0]?.metrics || {}),
            ].filter(Boolean),
        ),
    ) as string[];
    const primary = query.get("metric") || metricNames[0] || "";
    const navigate = useNavigate();
    function filter(key: string, value: string) {
        const next = new URLSearchParams(query);
        next.set(key, value);
        next.delete("offset");
        setQuery(next);
    }
    return (
        <>
            <Heading
                title="Прогоны"
                sub="Выберите прогоны, изучите контракт и сравните результаты"
            >
                <button onClick={reload}>Обновить таблицу</button>
                <Link className="button" to="/sources">
                    Сканировать источники
                </Link>
                <button
                    className="primary"
                    disabled={selected.length < 2 || selected.length > 5}
                    onClick={() =>
                        navigate(
                            "/runs/compare?ids=" + selected.map(id).join(","),
                        )
                    }
                >
                    Сравнить ({selected.length})
                </button>
            </Heading>
            <div className="toolbar">
                <input
                    aria-label="Поиск прогонов"
                    placeholder="Поиск по имени или ID"
                    value={query.get("q") || ""}
                    onChange={(e) => filter("q", e.target.value)}
                />
                <select
                    aria-label="Источник"
                    value={query.get("source") || ""}
                    onChange={(e) => filter("source", e.target.value)}
                >
                    <option value="">Все источники</option>
                    {sources.data?.map((s) => (
                        <option key={s.id} value={s.id}>
                            {s.name}
                        </option>
                    ))}
                </select>
                <select
                    aria-label="Метрика таблицы"
                    value={primary}
                    onChange={(e) => filter("metric", e.target.value)}
                >
                    <option value="">Основная метрика</option>
                    {metricNames.map((m) => (
                        <option key={m}>{m}</option>
                    ))}
                </select>
                <select
                    aria-label="Статус прогона"
                    value={query.get("status") || ""}
                    onChange={(e) => filter("status", e.target.value)}
                >
                    <option value="">Все состояния</option>
                    {[
                        "finished",
                        "running",
                        "failed",
                        "invalid",
                        "unknown",
                    ].map((v) => (
                        <option key={v} value={v}>
                            {names[v] || v}
                        </option>
                    ))}
                </select>
                <select
                    aria-label="Сортировка"
                    value={query.get("sort") || "name"}
                    onChange={(e) => filter("sort", e.target.value)}
                >
                    <option value="name">По имени ↑</option>
                    <option value="-name">По имени ↓</option>
                    <option value="source">По источнику</option>
                    <option value="status">По статусу</option>
                </select>
            </div>
            {!data ? (
                <Wait error={error} />
            ) : data.total === 0 ? (
                <Empty>
                    <h2>Прогонов пока нет</h2>
                    <p>
                        Подключите каталог с результатами save_run_files() или
                        MLflow с log_contract_artifacts().
                    </p>
                    <Link className="button" to="/sources">
                        Подключить источник
                    </Link>
                </Empty>
            ) : (
                <section>
                    <div className="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Выбор</th>
                                    <th>Прогон</th>
                                    <th>Источник / группа</th>
                                    <th>Состояние источника</th>
                                    <th>Версия данных</th>
                                    <th>{primary || "Метрика"}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.items.map((r) => (
                                    <tr key={r.id}>
                                        <td>
                                            <input
                                                type="checkbox"
                                                aria-label={"Выбрать " + r.name}
                                                checked={selected.includes(
                                                    r.id,
                                                )}
                                                onChange={(e) =>
                                                    setSelected(
                                                        e.target.checked
                                                            ? [
                                                                  ...selected,
                                                                  r.id,
                                                              ]
                                                            : selected.filter(
                                                                  (x) =>
                                                                      x !==
                                                                      r.id,
                                                              ),
                                                    )
                                                }
                                            />
                                        </td>
                                        <td>
                                            <RunLink run={r} />
                                        </td>
                                        <td>
                                            {r.source_name}
                                            <small>{r.experiment}</small>
                                        </td>
                                        <td>
                                            <Badge value={r.status} />
                                        </td>
                                        <td>{r.data_version || "—"}</td>
                                        <td className="mono">
                                            {typeof r.metrics?.[primary] ===
                                            "object"
                                                ? show(
                                                      (
                                                          r.metrics[
                                                              primary
                                                          ] as any
                                                      ).value,
                                                  )
                                                : show(r.metrics?.[primary])}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <p className="muted">
                        Для файлов значения метрик появляются после открытия
                        карточки прогона. Прочерк означает отсутствие расчёта.
                    </p>
                    <Pager
                        total={data.total}
                        offset={Number(query.get("offset") || 0)}
                        onChange={(offset) => {
                            const next = new URLSearchParams(query);
                            next.set("offset", String(offset));
                            setQuery(next);
                        }}
                    />
                </section>
            )}
        </>
    );
}

function RunDetail() {
    const { runId = "" } = useParams(),
        [query, setQuery] = useSearchParams(),
        { data, error, reload } = useData<Run>("/runs/" + id(runId)),
        [actionError, setError] = useState(""),
        [checked, setChecked] = useState<any>();
    const tab = query.get("tab") || "summary";
    return (
        <>
            <Heading title={data?.name || "Прогон"} sub={runId}>
                <Link
                    className="button primary"
                    to={"/runs/" + id(runId) + "/review"}
                >
                    Ревью прогона →
                </Link>
                <button
                    onClick={async () => {
                        try {
                            setChecked(
                                await api(
                                    "/runs/" + id(runId) + "/contract-check",
                                    "POST",
                                ),
                            );
                            setError("");
                        } catch (e) {
                            setError((e as Error).message);
                        }
                    }}
                >
                    Проверить контракт
                </button>
                <button onClick={reload}>Обновить</button>
            </Heading>
            <ErrorBox text={actionError} />
            {checked && (
                <section>
                    <h2>Проверка контракта</h2>
                    <Contract value={checked} />
                </section>
            )}
            <div className="tabs">
                {[
                    ["summary", "Сводка"],
                    ["metrics", "Метрики и кривые"],
                    ["parameters", "Параметры и модель"],
                    ["contract", "Данные и контракт"],
                    ["findings", "Находки"],
                    ["artifacts", "Артефакты"],
                ].map(([key, label]) => (
                    <button
                        key={key}
                        className={tab === key ? "active" : ""}
                        onClick={() => setQuery({ tab: key })}
                    >
                        {label}
                    </button>
                ))}
            </div>
            {!data ? (
                <Wait error={error} />
            ) : (
                <section>
                    {tab === "summary" ? (
                        <>
                            <div className="row">
                                <Badge value={data.status} />
                                <span>
                                    Версия данных:{" "}
                                    {data.data_version || "не задана"}
                                </span>
                            </div>
                            <h2>Метрики</h2>
                            <Metrics run={data} />
                            <h2>Контракт</h2>
                            <Contract value={data.contract} />
                        </>
                    ) : tab === "metrics" ? (
                        <>
                            <Metrics run={data} />
                            <h2>Кривые обучения</h2>
                            <Curves rows={data.curves || []} />
                        </>
                    ) : tab === "parameters" ? (
                        <>
                            <h2>Параметры</h2>
                            <Json value={data.parameters || {}} />
                            <h2>Описание модели</h2>
                            <Json value={data.model || "Нет metadata модели"} />
                        </>
                    ) : tab === "contract" ? (
                        <Contract value={data.contract} />
                    ) : tab === "findings" ? (
                        <Findings embeddedRun={runId} />
                    ) : (
                        <p>{data.artifacts || "Артефакты недоступны"}</p>
                    )}
                </section>
            )}
        </>
    );
}
function Contract({ value }: { value: any }) {
    return (
        <>
            <Badge value={value?.status || "unknown"} />
            {value?.error && <ErrorBox text={value.error} />}
            <dl>
                {value?.gaps?.map((g: any) => (
                    <React.Fragment key={g.field + g.purpose}>
                        <dt>{g.field}</dt>
                        <dd>{g.purpose}</dd>
                    </React.Fragment>
                ))}
            </dl>
            <p className="muted">
                Применимость отдельных анализаторов и точные причины пропусков
                доступны в отчёте ревью.
            </p>
        </>
    );
}
function Metrics({ run }: { run: Run }) {
    return run.metrics && Object.keys(run.metrics).length ? (
        <table>
            <thead>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Направление</th>
                </tr>
            </thead>
            <tbody>
                {Object.entries(run.metrics).map(([name, value]) => (
                    <tr key={name}>
                        <td>{name}</td>
                        <td className="mono">
                            {show(
                                typeof value === "object"
                                    ? value?.value
                                    : value,
                            )}
                        </td>
                        <td>
                            {typeof value === "object"
                                ? value?.higher_is_better
                                    ? "Больше лучше"
                                    : "Меньше лучше"
                                : "—"}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    ) : (
        <p className="muted">
            Метрики недоступны. Проверьте спецификацию и контракт.
        </p>
    );
}
function Curves({ rows }: { rows: Record<string, unknown>[] }) {
    if (!rows.length)
        return <p className="muted">Кривые отсутствуют в контракте.</p>;
    return (
        <>
            {Array.from(new Set(rows.map((r) => String(r.name)))).map(
                (name) => {
                    const points = rows
                        .filter(
                            (r) =>
                                String(r.name) === name &&
                                Number.isFinite(Number(r.value)),
                        )
                        .sort((a, b) => Number(a.step) - Number(b.step));
                    const values = points.map((r) => Number(r.value)),
                        steps = points.map((r) => Number(r.step));
                    const lo = Math.min(...values),
                        hi = Math.max(...values),
                        first = Math.min(...steps),
                        last = Math.max(...steps);
                    return (
                        <figure key={name}>
                            <figcaption>
                                {name} · шаги {first}–{last} · значения {lo}–
                                {hi}
                            </figcaption>
                            <svg
                                viewBox="0 0 800 200"
                                role="img"
                                aria-label={"Кривая " + name}
                            >
                                <polyline
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    points={points
                                        .map(
                                            (r) =>
                                                `${20 + ((Number(r.step) - first) * 760) / (last - first || 1)},${180 - ((Number(r.value) - lo) * 160) / (hi - lo || 1)}`,
                                        )
                                        .join(" ")}
                                />
                            </svg>
                        </figure>
                    );
                },
            )}
            <details>
                <summary>Точные данные кривых</summary>
                <Json value={rows} />
            </details>
        </>
    );
}

function Review() {
    const { runId = "" } = useParams(),
        [query, setQuery] = useSearchParams();
    const { data, error, reload } = useData<any>(
        "/runs/" +
            id(runId) +
            "/review" +
            (query.get("context")
                ? "?context=" + id(query.get("context")!)
                : ""),
    );
    const [failure, setFailure] = useState(""),
        [groups, setGroups] = useState<string[]>([]);
    const jobId = query.get("job");
    async function start() {
        try {
            const job = await api<Job>("/reviews", "POST", {
                run_id: runId,
                groups,
            });
            setQuery({ job: job.id });
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    return (
        <>
            <Heading title="Ревью прогона" sub={runId}>
                <button className="primary" onClick={start}>
                    Запустить ревью
                </button>
                <Link className="button" to={"/runs/" + id(runId)}>
                    Открыть прогон
                </Link>
            </Heading>
            <ErrorBox text={failure} />
            <section>
                <h2>Группы анализаторов</h2>
                <p className="muted">
                    Без выбора запускаются все группы. Числа и выводы
                    рассчитывает детерминированное ядро.
                </p>
                <div className="actions">
                    {[
                        "data",
                        "segments",
                        "metrics",
                        "model",
                        "curves",
                        "process",
                    ].map((g) => (
                        <label key={g}>
                            <input
                                type="checkbox"
                                checked={groups.includes(g)}
                                onChange={(e) =>
                                    setGroups(
                                        e.target.checked
                                            ? [...groups, g]
                                            : groups.filter((x) => x !== g),
                                    )
                                }
                            />
                            {g}
                        </label>
                    ))}
                </div>
            </section>
            {jobId && (
                <JobStatus
                    jobId={jobId}
                    onDone={(job) => {
                        reload();
                        if (job.result?.context)
                            setQuery({ context: job.result.context });
                    }}
                />
            )}
            {!data ? (
                <Wait error={error} />
            ) : data.status === "not_run" ? (
                <Empty>
                    <h2>Ревью ещё не запускалось</h2>
                    <p>{data.explanation}</p>
                </Empty>
            ) : (
                <Report data={data} />
            )}
        </>
    );
}
function Report({ data }: { data: any }) {
    const [error, setError] = useState(""),
        [severity, setSeverity] = useState("");
    const report = data.report;
    async function copy() {
        try {
            await navigator.clipboard.writeText(data.cli);
        } catch {
            setError("Скопируйте команду из блока ниже");
        }
    }
    function download() {
        const url = URL.createObjectURL(
            new Blob([JSON.stringify(report, null, 2)], {
                type: "application/json",
            }),
        );
        const a = document.createElement("a");
        a.href = url;
        a.download = "mlreview-report.json";
        a.click();
        URL.revokeObjectURL(url);
    }
    return (
        <>
            <section>
                <div className="row">
                    <h2>
                        Отчёт <Badge value={data.status} />
                    </h2>
                    <div className="actions">
                        <button onClick={download}>Скачать JSON</button>
                        <button onClick={copy}>Скопировать CLI</button>
                    </div>
                </div>
                <p className="muted">
                    {new Date(data.created_at).toLocaleString(undefined, {
                        timeZoneName: "short",
                    })}{" "}
                    · версия данных {report.data_version || "не задана"}
                </p>
                {data.stale && (
                    <div className="warning">
                        Входные данные или спецификация изменились. Запустите
                        ревью заново.
                    </div>
                )}
                <p>
                    Базовый прогон: {report.baseline_id || "не выбран"} ·
                    группы: {data.request?.groups?.join(", ") || "все"}
                </p>
                <details>
                    <summary>Контекст и воспроизводимость</summary>
                    <Json
                        value={{
                            ...data.request,
                            task_hash: data.task_hash,
                            analysis_version: data.analysis_version,
                        }}
                    />
                </details>
                <ErrorBox text={error} />
                <details>
                    <summary>Эквивалент CLI</summary>
                    <pre>{data.cli}</pre>
                </details>
                <label>
                    Важность{" "}
                    <select
                        value={severity}
                        onChange={(e) => setSeverity(e.target.value)}
                    >
                        <option value="">Все</option>
                        {["critical", "high", "medium", "low", "info"].map(
                            (s) => (
                                <option key={s}>{s}</option>
                            ),
                        )}
                    </select>
                </label>
                <FindingsTable
                    items={(report.findings || []).filter(
                        (f: Finding) => !severity || f.severity === severity,
                    )}
                />
                {!report.findings?.length && (
                    <p>
                        Ревью выполнено: показанных находок нет. Проверьте
                        список невыполненных проверок ниже.
                    </p>
                )}
            </section>
            <section>
                <h2>Что не проверено · {report.skipped?.length || 0}</h2>
                {report.skipped?.length ? (
                    <dl>
                        {report.skipped.map((s: any, i: number) => (
                            <React.Fragment key={i}>
                                <dt>{s.check}</dt>
                                <dd>{s.reason}</dd>
                            </React.Fragment>
                        ))}
                    </dl>
                ) : (
                    <p>Пропусков в отчёте нет.</p>
                )}
            </section>
            <div className="notice">
                LLM-гипотезы доступны в следующем этапе UI. Этот отчёт содержит
                детерминированные факты.
            </div>
        </>
    );
}

function Compare() {
    const [query, setQuery] = useSearchParams();
    const ids = (query.get("ids") || "").split(",").filter(Boolean);
    const baseline = query.get("baseline") || "";
    const [runs, setRuns] = useState<Run[]>([]),
        [error, setError] = useState(""),
        [jobs, setJobs] = useState<Job[]>([]),
        [grouped, setGrouped] = useState(false),
        [baselineSeeds, setBaselineSeeds] = useState<string[]>([]);
    useEffect(() => {
        Promise.all(ids.map((r) => api<Run>("/runs/" + id(r))))
            .then(setRuns)
            .catch((e) => setError(e.message));
    }, [query.get("ids")]);
    async function compare() {
        try {
            const candidates = ids.filter(
                (r) => r !== baseline && !baselineSeeds.includes(r),
            );
            if (!candidates.length)
                throw new Error("Выберите хотя бы один прогон кандидата");
            setJobs(
                grouped
                    ? [
                          await api<Job>("/reviews", "POST", {
                              run_id: candidates[0],
                              seeds: candidates.slice(1),
                              baseline,
                              baseline_seeds: baselineSeeds,
                          }),
                      ]
                    : await Promise.all(
                          candidates.map((run) =>
                              api<Job>("/reviews", "POST", {
                                  run_id: run,
                                  baseline,
                              }),
                          ),
                      ),
            );
            setError("");
        } catch (e) {
            setError((e as Error).message);
        }
    }
    return (
        <>
            <Heading
                title="Сравнение прогонов"
                sub="Явный baseline и результаты существующих анализаторов"
            >
                <button
                    className="primary"
                    disabled={!baseline || ids.length < 2 || ids.length > 5}
                    onClick={compare}
                >
                    Выполнить сравнение
                </button>
            </Heading>
            <ErrorBox text={error} />
            <section>
                <label>
                    Базовый прогон{" "}
                    <select
                        value={baseline}
                        onChange={(e) => {
                            const next = new URLSearchParams(query);
                            next.set("baseline", e.target.value);
                            setQuery(next);
                        }}
                    >
                        <option value="">Выберите baseline</option>
                        {runs.map((r) => (
                            <option key={r.id} value={r.id}>
                                {r.name}
                            </option>
                        ))}
                    </select>
                </label>
                <label>
                    <input
                        type="checkbox"
                        checked={grouped}
                        onChange={(e) => {
                            setGrouped(e.target.checked);
                            setBaselineSeeds([]);
                        }}
                    />
                    Сравнить группы сидов
                </label>
                {grouped &&
                    runs
                        .filter((r) => r.id !== baseline)
                        .map((r) => (
                            <label key={r.id}>
                                <input
                                    type="checkbox"
                                    checked={baselineSeeds.includes(r.id)}
                                    onChange={(e) =>
                                        setBaselineSeeds(
                                            e.target.checked
                                                ? [...baselineSeeds, r.id]
                                                : baselineSeeds.filter(
                                                      (x) => x !== r.id,
                                                  ),
                                        )
                                    }
                                />
                                {r.name}: сид baseline (остальные — группа
                                кандидата)
                            </label>
                        ))}
                {ids.length < 2 || ids.length > 5 ? (
                    <ErrorBox text="Выберите от 2 до 5 прогонов в таблице" />
                ) : null}
                {new Set(runs.map((r) => r.data_version)).size > 1 && (
                    <div className="warning">
                        Версии данных различаются. Сопоставимость и пропуски
                        определяются ядром.
                    </div>
                )}
                <div className="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Показатель</th>
                                {runs.map((r) => (
                                    <th key={r.id}>
                                        <RunLink run={r} />
                                        {r.id === baseline && (
                                            <small>БАЗОВЫЙ</small>
                                        )}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th>Версия данных</th>
                                {runs.map((r) => (
                                    <td key={r.id}>
                                        {r.data_version || "не задана"}
                                    </td>
                                ))}
                            </tr>
                            {Array.from(
                                new Set(
                                    runs.flatMap((r) =>
                                        Object.keys(r.metrics || {}),
                                    ),
                                ),
                            ).map((m) => (
                                <tr key={m}>
                                    <th>{m}</th>
                                    {runs.map((r) => (
                                        <td key={r.id}>
                                            {show(
                                                typeof r.metrics?.[m] ===
                                                    "object"
                                                    ? (r.metrics[m] as any)
                                                          .value
                                                    : r.metrics?.[m],
                                            )}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                            {Array.from(
                                new Set(
                                    runs.flatMap((r) =>
                                        Object.keys(r.parameters || {}),
                                    ),
                                ),
                            ).map((p) => (
                                <tr key={p}>
                                    <th>{p}</th>
                                    {runs.map((r) => (
                                        <td key={r.id}>
                                            {show(r.parameters?.[p])}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <p className="muted">
                    Δ, доверительные интервалы и сегментные регрессии появятся в
                    отчётах анализаторов после сравнения.
                </p>
            </section>
            {jobs.map((j) => (
                <section key={j.id}>
                    <JobStatus jobId={j.id} />
                    {j.result?.run_id && (
                        <Link to={"/runs/" + id(j.result.run_id) + "/review"}>
                            Открыть отчёт
                        </Link>
                    )}
                    <Link
                        to={
                            "/runs/" +
                            id(
                                (j as any).request?.run_id ||
                                    ids.find((r) => r !== baseline) ||
                                    "",
                            ) +
                            "/review?job=" +
                            j.id
                        }
                    >
                        Открыть отчёт сравнения →
                    </Link>
                </section>
            ))}
        </>
    );
}

function Findings({ embeddedRun }: { embeddedRun?: string }) {
    const [query, setQuery] = useSearchParams();
    const path =
        "/findings?" +
        (embeddedRun ? "run=" + id(embeddedRun) : query.toString());
    const { data, error } = useData<Page<Finding>>(path);
    function filter(key: string, value: string) {
        const next = new URLSearchParams(query);
        next.set(key, value);
        next.delete("offset");
        setQuery(next);
    }
    return (
        <>
            {!embeddedRun && (
                <Heading
                    title="Находки"
                    sub="Общий backlog CLI и Web. Факты, evidence и решения."
                />
            )}
            <div className="toolbar">
                <input
                    aria-label="Поиск находок"
                    placeholder="Поиск по тексту"
                    value={query.get("q") || ""}
                    onChange={(e) => filter("q", e.target.value)}
                />
                <select
                    aria-label="Статус находки"
                    value={query.get("status") || ""}
                    onChange={(e) => filter("status", e.target.value)}
                >
                    <option value="">Все статусы</option>
                    {["new", "accepted", "dismissed", "fixed"].map((s) => (
                        <option key={s} value={s}>
                            {names[s]}
                        </option>
                    ))}
                </select>
                <select
                    aria-label="Важность"
                    value={query.get("severity") || ""}
                    onChange={(e) => filter("severity", e.target.value)}
                >
                    <option value="">Вся важность</option>
                    {["critical", "high", "medium", "low", "info"].map((s) => (
                        <option key={s} value={s}>
                            {names[s]}
                        </option>
                    ))}
                </select>
            </div>
            {!data ? (
                <Wait error={error} />
            ) : (
                <section>
                    <FindingsTable items={data.items || []} />
                    <Pager
                        total={data.total || 0}
                        offset={Number(query.get("offset") || 0)}
                        onChange={(n) => {
                            const next = new URLSearchParams(query);
                            next.set("offset", String(n));
                            setQuery(next);
                        }}
                    />
                </section>
            )}
        </>
    );
}
function FindingDetail() {
    const { findingId = "" } = useParams(),
        { data, error, reload } = useData<Finding>(
            "/findings/" + id(findingId),
        ),
        [reason, setReason] = useState(""),
        [failure, setFailure] = useState("");
    async function status(value: string) {
        try {
            await api("/findings/" + id(findingId) + "/status", "POST", {
                status: value,
                reason,
            });
            reload();
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    return (
        <>
            <Heading
                title={"Находка " + findingId}
                sub="Факты и evidence из общего хранилища"
            >
                <Link
                    className="button"
                    to={"/journal?finding=" + id(findingId)}
                >
                    Добавить решение
                </Link>
            </Heading>
            {!data ? (
                <Wait error={error} />
            ) : (
                <>
                    <section>
                        <div className="actions">
                            <Badge value={data.severity} />
                            <Badge value={data.status} />
                            <span className="mono">{data.check}</span>
                        </div>
                        <h2>{data.message}</h2>
                        {data.reason && <p>Причина: {data.reason}</p>}
                        <h3>Наблюдаемые факты</h3>
                        <Json value={data.facts} />
                        <h3>Evidence</h3>
                        {data.evidence?.map((s, i) => (
                            <p className="mono" key={i}>
                                {s}
                            </p>
                        ))}
                        <h3>Следующий эксперимент</h3>
                        <p>{data.suggestion || "Рекомендация отсутствует"}</p>
                        {data.run_ids?.map((r) => (
                            <Link
                                className="button"
                                key={r}
                                to={"/runs/" + id(r)}
                            >
                                Открыть прогон →
                            </Link>
                        ))}
                    </section>
                    <section>
                        <h2>Решение по находке</h2>
                        <textarea
                            aria-label="Причина изменения статуса"
                            placeholder="Причина (обязательна при отклонении critical)"
                            value={reason}
                            onChange={(e) => setReason(e.target.value)}
                        />
                        <div className="actions">
                            <button onClick={() => status("accepted")}>
                                Принять
                            </button>
                            <button onClick={() => status("dismissed")}>
                                Подтвердить отклонение
                            </button>
                            <button onClick={() => status("new")}>
                                Вернуть в новые
                            </button>
                        </div>
                        <ErrorBox text={failure} />
                        <p className="muted">
                            Отклонение не означает исправление. Повторное
                            обнаружение обрабатывается правилами ядра.
                        </p>
                    </section>
                    <section>
                        <h2>История</h2>
                        {data.history?.map((h, i) => (
                            <div className="row" key={i}>
                                <span>
                                    {new Date(h.at).toLocaleString(undefined, {
                                        timeZoneName: "short",
                                    })}
                                </span>
                                <Badge value={h.status} />
                                <span>{h.reason}</span>
                            </div>
                        ))}
                    </section>
                </>
            )}
        </>
    );
}

function Journal() {
    const [query, setQuery] = useSearchParams(),
        { data, error, reload } = useData<Entry[]>("/journal?" + query),
        [text, setText] = useState(""),
        [kind, setKind] = useState("note"),
        [run, setRun] = useState(""),
        [outcome, setOutcome] = useState(""),
        [hypothesis, setHypothesis] = useState(""),
        [failure, setFailure] = useState("");
    async function add(e: React.FormEvent) {
        e.preventDefault();
        try {
            await api("/journal", "POST", {
                kind,
                text,
                runs: run ? [run] : [],
                findings: query.get("finding") ? [query.get("finding")] : [],
                hypothesis: hypothesis || null,
                outcome: outcome || null,
            });
            setText("");
            reload();
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    return (
        <>
            <Heading
                title="Журнал"
                sub="Гипотезы, наблюдения и принятые решения"
            />
            <div className="toolbar">
                <input
                    aria-label="Поиск в журнале"
                    placeholder="Найти прошлое решение"
                    value={query.get("q") || ""}
                    onChange={(e) => setQuery({ q: e.target.value })}
                />
            </div>
            <section>
                <h2>Новая запись</h2>
                <form onSubmit={add}>
                    <div className="actions">
                        <label>
                            Тип{" "}
                            <select
                                value={kind}
                                onChange={(e) => {
                                    setKind(e.target.value);
                                    setOutcome("");
                                    setHypothesis("");
                                }}
                            >
                                <option value="note">Заметка</option>
                                <option value="hypothesis">Гипотеза</option>
                                <option value="decision">Решение</option>
                            </select>
                        </label>
                        <label>
                            Прогон (ID)
                            <input
                                value={run}
                                onChange={(e) => setRun(e.target.value)}
                            />
                        </label>
                        {kind === "decision" && (
                            <>
                                <label>
                                    Исход
                                    <select
                                        value={outcome}
                                        onChange={(e) =>
                                            setOutcome(e.target.value)
                                        }
                                    >
                                        <option value="">Не указан</option>
                                        <option value="accept">Принять</option>
                                        <option value="reject">
                                            Отклонить
                                        </option>
                                    </select>
                                </label>
                                <label>
                                    Гипотеза (H-…)
                                    <input
                                        value={hypothesis}
                                        onChange={(e) =>
                                            setHypothesis(e.target.value)
                                        }
                                    />
                                </label>
                            </>
                        )}
                    </div>
                    <textarea
                        required
                        aria-label="Текст записи"
                        placeholder="Что проверили и какой вывод сделали?"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />
                    <button className="primary">Добавить запись</button>
                    <ErrorBox text={failure} />
                </form>
            </section>
            {!data ? (
                <Wait error={error} />
            ) : (
                <section>
                    <h2>Память проекта</h2>
                    {data.length ? (
                        data.map((e) => (
                            <article key={e.id}>
                                <p className="eyebrow">
                                    {e.id} · {names[e.kind] || e.kind} ·{" "}
                                    {new Date(e.created_at).toLocaleString(
                                        undefined,
                                        { timeZoneName: "short" },
                                    )}
                                </p>
                                <Link to={"/journal/" + e.id}>{e.text}</Link>
                                <small>
                                    {e.outcome || ""} {e.runs?.join(", ")}
                                </small>
                            </article>
                        ))
                    ) : (
                        <p className="muted">Записей пока нет.</p>
                    )}
                </section>
            )}
        </>
    );
}
function JournalDetail() {
    const { entryId = "" } = useParams(),
        { data, error } = useData<Entry>("/journal/" + id(entryId));
    return (
        <>
            <Heading title={"Запись " + entryId} />
            {data ? (
                <section>
                    <p>{data.text}</p>
                    <Json value={data} />
                    {data.findings.map((f) => (
                        <Link key={f} to={"/findings/" + id(f)}>
                            {f} →{" "}
                        </Link>
                    ))}
                </section>
            ) : (
                <Wait error={error} />
            )}
        </>
    );
}

function TaskEditor() {
    const { data, error, reload } = useData<Task>("/task"),
        [text, setText] = useState(""),
        [failure, setFailure] = useState(""),
        [message, setMessage] = useState(""),
        [preview, setPreview] = useState(false);
    useEffect(() => {
        if (data)
            setText(
                data.text ||
                    "task:\n  type: classification/binary\n  output: probabilities\n",
            );
    }, [data]);
    let document: any;
    try {
        document = parse(text);
    } catch {
        document = null;
    }
    function field(section: string, key: string, value: unknown) {
        setText(
            stringify({
                ...document,
                [section]: { ...document?.[section], [key]: value },
            }),
        );
        setPreview(false);
    }
    async function validate() {
        try {
            await api("/task/validate", "POST", { text });
            setMessage("Спецификация валидна");
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    async function save() {
        try {
            await api("/task", "PUT", { text, etag: data?.etag });
            setMessage("Сохранено с резервной копией");
            setPreview(false);
            reload();
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    return (
        <>
            <Heading title="Спецификация задачи" sub={data?.path}>
                <button onClick={validate}>Проверить</button>
                <button onClick={() => setPreview(true)}>
                    Сравнить изменения
                </button>
                <button className="primary" disabled={!preview} onClick={save}>
                    Подтвердить сохранение
                </button>
            </Heading>
            <ErrorBox text={failure} />
            {message && (
                <div className="notice" role="status">
                    {message}
                </div>
            )}
            {!data ? (
                <Wait error={error} />
            ) : (
                <>
                    <div className="grid">
                        <section>
                            <h2>Задача</h2>
                            <label>
                                Тип
                                <select
                                    value={document?.task?.type || ""}
                                    onChange={(e) =>
                                        field("task", "type", e.target.value)
                                    }
                                >
                                    <option value="classification/binary">
                                        Бинарная классификация
                                    </option>
                                    <option value="classification/multiclass">
                                        Многоклассовая классификация
                                    </option>
                                    <option value="regression">
                                        Регрессия
                                    </option>
                                </select>
                            </label>
                            <label>
                                Выход
                                <select
                                    value={document?.task?.output || ""}
                                    onChange={(e) =>
                                        field("task", "output", e.target.value)
                                    }
                                >
                                    {[
                                        "probabilities",
                                        "labels",
                                        "scores",
                                        "values",
                                    ].map((v) => (
                                        <option key={v}>{v}</option>
                                    ))}
                                </select>
                            </label>
                            <label>
                                Сегменты через запятую
                                <input
                                    value={
                                        document?.task?.segments?.join(", ") ||
                                        ""
                                    }
                                    onChange={(e) =>
                                        field(
                                            "task",
                                            "segments",
                                            e.target.value
                                                .split(",")
                                                .map((v) => v.trim())
                                                .filter(Boolean),
                                        )
                                    }
                                />
                            </label>
                            <label>
                                Версия данных
                                <input
                                    value={document?.data?.version || ""}
                                    onChange={(e) =>
                                        field("data", "version", e.target.value)
                                    }
                                />
                            </label>
                            <label>
                                Стратегия разбиения
                                <input
                                    value={document?.data?.split_strategy || ""}
                                    onChange={(e) =>
                                        field(
                                            "data",
                                            "split_strategy",
                                            e.target.value || null,
                                        )
                                    }
                                />
                            </label>
                            <label>
                                Колонка времени
                                <input
                                    value={document?.data?.time_column || ""}
                                    onChange={(e) =>
                                        field(
                                            "data",
                                            "time_column",
                                            e.target.value || null,
                                        )
                                    }
                                />
                            </label>
                            <label>
                                Известные проблемы (по одной в строке)
                                <textarea
                                    value={
                                        document?.data?.known_issues?.join(
                                            "\n",
                                        ) || ""
                                    }
                                    onChange={(e) =>
                                        field(
                                            "data",
                                            "known_issues",
                                            e.target.value.split("\n"),
                                        )
                                    }
                                />
                            </label>
                            <h3>Стоимость ошибок</h3>
                            {(document?.task?.type === "regression"
                                ? ["underestimation", "overestimation"]
                                : ["false_positive", "false_negative"]
                            ).map((cost) => (
                                <label key={cost}>
                                    {cost}
                                    <select
                                        value={
                                            document?.task?.error_costs?.[
                                                cost
                                            ] || ""
                                        }
                                        onChange={(e) =>
                                            field("task", "error_costs", {
                                                ...document?.task?.error_costs,
                                                [cost]: e.target.value,
                                            })
                                        }
                                    >
                                        <option value="">Не задана</option>
                                        {["low", "medium", "high"].map(
                                            (level) => (
                                                <option key={level}>
                                                    {level}
                                                </option>
                                            ),
                                        )}
                                    </select>
                                </label>
                            ))}
                            <h3>Метрики</h3>
                            <label>
                                Основная метрика
                                <input
                                    value={document?.metrics?.primary || ""}
                                    onChange={(e) =>
                                        field(
                                            "metrics",
                                            "primary",
                                            e.target.value,
                                        )
                                    }
                                    placeholder="roc_auc / mae / …"
                                />
                            </label>
                            <label>
                                Дополнительные метрики
                                <input
                                    value={
                                        document?.metrics?.secondary?.join(
                                            ", ",
                                        ) || ""
                                    }
                                    onChange={(e) =>
                                        field(
                                            "metrics",
                                            "secondary",
                                            e.target.value
                                                .split(",")
                                                .map((v) => v.trim())
                                                .filter(Boolean),
                                        )
                                    }
                                />
                            </label>
                            <label>
                                Порог классификации
                                <input
                                    type="number"
                                    step="0.01"
                                    value={document?.metrics?.threshold ?? 0.5}
                                    onChange={(e) =>
                                        field(
                                            "metrics",
                                            "threshold",
                                            Number(e.target.value),
                                        )
                                    }
                                />
                            </label>
                            <p className="muted">
                                Дополнительные параметры доступны в YAML.
                                Валидацию всех полей выполняет Python.
                            </p>
                        </section>
                        <section>
                            <h2>YAML</h2>
                            <textarea
                                className="yaml"
                                aria-label="YAML задачи"
                                value={text}
                                onChange={(e) => {
                                    setText(e.target.value);
                                    setPreview(false);
                                }}
                                spellCheck={false}
                            />
                        </section>
                    </div>
                    {preview && (
                        <section>
                            <h2>Проверьте изменения перед сохранением</h2>
                            <div className="grid">
                                <div>
                                    <h3>На диске</h3>
                                    <pre>
                                        {data.text || "(файл отсутствует)"}
                                    </pre>
                                </div>
                                <div>
                                    <h3>После сохранения</h3>
                                    <pre>{text}</pre>
                                </div>
                            </div>
                        </section>
                    )}
                </>
            )}
        </>
    );
}

function Sources({ setup = false }: { setup?: boolean }) {
    const { data, error, reload } = useData<Source[]>("/sources"),
        [kind, setKind] = useState("files"),
        [name, setName] = useState("Локальные прогоны"),
        [path, setPath] = useState("./runs"),
        [uri, setUri] = useState(""),
        [experiments, setExperiments] = useState<
            { id: string; name: string }[]
        >([]),
        [selected, setSelected] = useState<string[]>([]),
        [failure, setFailure] = useState(""),
        [jobs, setJobs] = useState<string[]>([]),
        [taskPath, setTaskPath] = useState("task.yaml"),
        [message, setMessage] = useState(""),
        [editId, setEditId] = useState("");
    const config = {
        name,
        kind,
        path,
        tracking_uri: uri,
        experiments: selected,
    };
    async function test() {
        try {
            const result = await api<any>("/sources/test", "POST", config);
            setExperiments(result.experiments || []);
            setMessage("Источник доступен");
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    async function add(e: React.FormEvent) {
        e.preventDefault();
        try {
            const result = await api<any>(
                editId ? "/sources/" + id(editId) : "/sources",
                editId ? "PUT" : "POST",
                config,
            );
            setJobs([...jobs, result.job.id]);
            setEditId("");
            reload();
            setFailure("");
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    async function refresh(source: Source) {
        try {
            const job = await api<Job>(
                "/sources/" + id(source.id) + "/refresh",
                "POST",
            );
            setJobs([...jobs, job.id]);
        } catch (e) {
            setFailure((e as Error).message);
        }
    }
    return (
        <>
            <Heading
                title={setup ? "Настройка проекта" : "Источники данных"}
                sub="Подключения хранятся локально. Обновление индекса не запускает ревью."
            >
                {setup && (
                    <Link className="button primary" to="/overview">
                        Завершить настройку →
                    </Link>
                )}
            </Heading>
            {setup && (
                <section>
                    <p className="eyebrow">ШАГ 1 / СПЕЦИФИКАЦИЯ</p>
                    <h2>Выберите файл задачи</h2>
                    <label>
                        Путь к task.yaml
                        <input
                            value={taskPath}
                            onChange={(e) => setTaskPath(e.target.value)}
                        />
                    </label>
                    <button
                        onClick={async () => {
                            try {
                                await api("/task/select", "POST", {
                                    path: taskPath,
                                });
                                setMessage("Файл выбран");
                                setFailure("");
                            } catch (e) {
                                setFailure((e as Error).message);
                            }
                        }}
                    >
                        Выбрать этот файл
                    </button>{" "}
                    <Link to="/task">Создать или отредактировать →</Link>
                    <p className="muted">
                        Этот шаг можно пропустить; ревью станет доступно после
                        настройки задачи.
                    </p>
                </section>
            )}
            <ErrorBox text={failure} />
            {message && <div className="notice">{message}</div>}
            <section>
                <p className="eyebrow">{setup ? "ШАГ 2 / " : ""}ПОДКЛЮЧЕНИЕ</p>
                <h2>
                    {editId ? "Редактирование источника" : "Добавить источник"}
                </h2>
                <form onSubmit={add}>
                    <div className="grid">
                        <div>
                            <label>
                                Название
                                <input
                                    required
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                />
                            </label>
                            <label>
                                Тип
                                <select
                                    value={kind}
                                    onChange={(e) => setKind(e.target.value)}
                                >
                                    <option value="files">
                                        Локальные файлы
                                    </option>
                                    <option value="mlflow">MLflow</option>
                                </select>
                            </label>
                        </div>
                        <div>
                            {kind === "files" ? (
                                <label>
                                    Каталог прогонов
                                    <input
                                        required
                                        value={path}
                                        onChange={(e) =>
                                            setPath(e.target.value)
                                        }
                                    />
                                </label>
                            ) : (
                                <>
                                    <label>
                                        Tracking URI
                                        <input
                                            required
                                            value={uri}
                                            onChange={(e) =>
                                                setUri(e.target.value)
                                            }
                                            placeholder="file:///… или http://…"
                                        />
                                    </label>
                                    {experiments.map((exp) => (
                                        <label key={exp.id}>
                                            <input
                                                type="checkbox"
                                                checked={selected.includes(
                                                    exp.id,
                                                )}
                                                onChange={(e) =>
                                                    setSelected(
                                                        e.target.checked
                                                            ? [
                                                                  ...selected,
                                                                  exp.id,
                                                              ]
                                                            : selected.filter(
                                                                  (x) =>
                                                                      x !==
                                                                      exp.id,
                                                              ),
                                                    )
                                                }
                                            />
                                            {exp.name}
                                        </label>
                                    ))}
                                </>
                            )}
                            <p className="muted">
                                Добавление явно разрешает чтение выбранного
                                каталога. Секреты задаются в окружении сервера.
                            </p>
                        </div>
                    </div>
                    <div className="actions">
                        <button type="button" onClick={test}>
                            Проверить подключение
                        </button>
                        <button
                            className="primary"
                            disabled={kind === "mlflow" && !selected.length}
                        >
                            {editId
                                ? "Сохранить и сканировать"
                                : "Подключить и сканировать"}
                        </button>
                        {editId && (
                            <button type="button" onClick={() => setEditId("")}>
                                Отменить
                            </button>
                        )}
                    </div>
                </form>
            </section>
            {jobs.map((job) => (
                <JobStatus key={job} jobId={job} onDone={reload} />
            ))}
            <section>
                <h2>Подключённые источники</h2>
                {!data ? (
                    <Wait error={error} />
                ) : data.length ? (
                    data.map((source) => (
                        <article key={source.id}>
                            <div className="row">
                                <h3>{source.name}</h3>
                                <div className="actions">
                                    <button
                                        onClick={() => {
                                            setEditId(source.id);
                                            setName(source.name);
                                            setKind(source.kind);
                                            setPath(source.path || "");
                                            setUri(source.tracking_uri || "");
                                            setSelected(
                                                source.experiments || [],
                                            );
                                            window.scrollTo({
                                                top: 0,
                                                behavior: "smooth",
                                            });
                                        }}
                                    >
                                        Изменить
                                    </button>
                                    <button onClick={() => refresh(source)}>
                                        Обновить индекс
                                    </button>
                                </div>
                            </div>
                            <p className="mono">
                                {source.kind === "files"
                                    ? source.path
                                    : source.tracking_uri}
                            </p>
                            <small>{source.id}</small>
                            <p>
                                {(source as any).available === false
                                    ? "⚠ Источник недоступен"
                                    : "Источник настроен"}{" "}
                                · Прогонов:{" "}
                                {(source as any).count ?? "не сканировалось"}
                            </p>
                            {(source as any).last_scan && (
                                <small>
                                    Последнее сканирование:{" "}
                                    {new Date(
                                        (source as any).last_scan,
                                    ).toLocaleString(undefined, {
                                        timeZoneName: "short",
                                    })}
                                </small>
                            )}
                            <ErrorBox text={(source as any).error || ""} />
                            <Link to={"/runs?source=" + id(source.id)}>
                                Открыть прогоны →
                            </Link>
                        </article>
                    ))
                ) : (
                    <p className="muted">Источников пока нет.</p>
                )}
            </section>
        </>
    );
}
function Settings() {
    const { data, error } = useData<any>("/settings"),
        jobs = useData<Job[]>("/jobs");
    return (
        <>
            <Heading
                title="Настройки и диагностика"
                sub="Локальное окружение MLReview"
            />
            {!data ? (
                <Wait error={error} />
            ) : (
                <>
                    <section>
                        <h2>Рабочее пространство</h2>
                        <dl>
                            {[
                                ["Проект", data.workspace],
                                ["SQLite", data.store],
                                ["Спецификация", data.task_path],
                                ["Версия", data.version],
                                ["Сервер", window.location.origin],
                            ].map(([label, value]) => (
                                <React.Fragment key={label}>
                                    <dt>{label}</dt>
                                    <dd className="mono">{value}</dd>
                                </React.Fragment>
                            ))}
                        </dl>
                    </section>
                    <div className="grid">
                        <section>
                            <h2>LLM</h2>
                            <p>Провайдер: {data.llm.provider}</p>
                            <p>
                                Ключ в окружении:{" "}
                                {data.llm.configured
                                    ? "настроен"
                                    : "не настроен"}
                            </p>
                            <p className="muted">{data.llm.explanation}</p>
                        </section>
                        <section>
                            <h2>Проактивный watch</h2>
                            <Badge value={data.watch.status} />
                            <p className="muted">{data.watch.explanation}</p>
                        </section>
                    </div>
                    <section>
                        <h2>Последние операции</h2>
                        {jobs.data?.map((j) => (
                            <div className="row" key={j.id}>
                                <span>{j.kind}</span>
                                <Badge value={j.status} />
                                <span>{j.error}</span>
                            </div>
                        ))}
                    </section>
                </>
            )}
        </>
    );
}

export default function App() {
    const [workspace, setWorkspace] = useState<Workspace>(),
        [error, setError] = useState("");
    const location = useLocation();
    useEffect(() => {
        api<Workspace>("/workspace")
            .then((w) => {
                setToken(w.csrf_token);
                setWorkspace(w);
            })
            .catch((e) => setError(e.message));
    }, []);
    if (!workspace)
        return (
            <div className="startup">
                <h1>MLReview</h1>
                <Wait error={error} />
            </div>
        );
    const project = workspace.root.split(/[\\/]/).filter(Boolean).pop();
    return (
        <div className="shell">
            <aside>
                <Link className="brand" to="/overview">
                    <span className="brand-icon">
                        M<span>↗</span>
                    </span>
                    MLReview<small>КРИТИК ЭКСПЕРИМЕНТОВ</small>
                </Link>
                <div className="project">
                    <small>ПРОЕКТ</small>
                    <strong>{project}</strong>
                    <span className="local">● Локально</span>
                </div>
                <nav aria-label="Разделы проекта">
                    {nav.map(([path, label, icon]) => (
                        <NavLink key={path} to={path}>
                            <span aria-hidden="true">{icon}</span>
                            {label}
                        </NavLink>
                    ))}
                </nav>
                <div className="sidebar-foot">
                    <span className="local">● Данные на вашей машине</span>
                    <small>MLReview {workspace.version}</small>
                </div>
            </aside>
            <div className="body">
                <header>
                    <span>
                        {project}{" "}
                        <span className="muted">
                            /{" "}
                            {nav.find(([p]) =>
                                location.pathname.startsWith(p),
                            )?.[1] || "Настройка"}
                        </span>
                    </span>
                    <span className="local">ЛОКАЛЬНЫЙ ПРОЕКТ</span>
                </header>
                <main key={location.pathname}>
                    <Routes>
                        <Route
                            path="/"
                            element={
                                <Navigate
                                    replace
                                    to={
                                        workspace.ready ? "/overview" : "/setup"
                                    }
                                />
                            }
                        />
                        <Route path="/overview" element={<Overview />} />
                        <Route path="/setup" element={<Sources setup />} />
                        <Route path="/sources" element={<Sources />} />
                        <Route path="/runs" element={<Runs />} />
                        <Route path="/runs/compare" element={<Compare />} />
                        <Route path="/runs/:runId" element={<RunDetail />} />
                        <Route
                            path="/runs/:runId/review"
                            element={<Review />}
                        />
                        <Route path="/findings" element={<Findings />} />
                        <Route
                            path="/findings/:findingId"
                            element={<FindingDetail />}
                        />
                        <Route path="/journal" element={<Journal />} />
                        <Route
                            path="/journal/:entryId"
                            element={<JournalDetail />}
                        />
                        <Route path="/task" element={<TaskEditor />} />
                        <Route path="/settings" element={<Settings />} />
                        <Route
                            path="*"
                            element={
                                <Empty>
                                    <h1>Страница не найдена</h1>
                                    <Link to="/runs">Вернуться к прогонам</Link>
                                </Empty>
                            }
                        />
                    </Routes>
                </main>
                <footer>
                    MLReview / Детерминированные факты. Осознанные решения.
                </footer>
            </div>
        </div>
    );
}
