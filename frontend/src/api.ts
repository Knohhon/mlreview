export type Workspace = {
    root: string;
    csrf_token: string;
    ready: boolean;
    version: string;
};
export type Source = {
    id: string;
    name: string;
    kind: "files" | "mlflow";
    path: string;
    tracking_uri: string;
    experiments: string[];
};
export type Run = {
    id: string;
    native_id: string;
    name: string;
    source: string;
    source_name: string;
    status: string;
    experiment: string;
    data_version?: string;
    metrics?: Record<
        string,
        number | { value: number | null; higher_is_better: boolean }
    >;
    parameters?: Record<string, unknown>;
    model?: Record<string, unknown>;
    contract?: {
        status: string;
        gaps?: { field: string; purpose: string }[];
        error?: string;
    };
    curves?: Record<string, unknown>[];
    artifacts?: string;
};
export type Finding = {
    id: string;
    severity: string;
    check: string;
    message: string;
    status: string;
    reason?: string;
    facts: Record<string, unknown>;
    evidence: string[];
    suggestion?: string;
    last_run?: string;
    run_ids?: string[];
    history?: { at: string; status: string; reason: string }[];
};
export type Job = {
    id: string;
    status: string;
    kind: string;
    error?: string;
    result?: { run_id?: string; context?: string; count?: number };
};
export type Page<T> = {
    items: T[];
    total: number;
    offset: number;
    limit: number;
};
export type Task = {
    text: string;
    etag: string;
    valid: boolean;
    error?: string;
    path: string;
    exists: boolean;
    parsed?: Record<string, any>;
};
export type Entry = {
    id: string;
    kind: string;
    text: string;
    runs: string[];
    findings: string[];
    outcome?: string;
    created_at: string;
};
let csrf = "";
export function setToken(value: string) {
    csrf = value;
}
export async function api<T>(
    path: string,
    method = "GET",
    data?: unknown,
): Promise<T> {
    const result = await fetch("/api/v1" + path, {
        method,
        headers:
            method === "GET"
                ? {}
                : { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: data === undefined ? undefined : JSON.stringify(data),
    });
    const payload = await result.json();
    if (!result.ok) throw new Error(payload.detail || "Операция не выполнена");
    return payload;
}
export const id = (value: string) => encodeURIComponent(value);
