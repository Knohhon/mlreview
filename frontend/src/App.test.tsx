import React from "react";
import { afterEach, expect, test, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";
afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
});
function mount(path: string) {
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => ({
            ok: true,
            json: async () =>
                url.endsWith("/workspace")
                    ? {
                          root: "/project",
                          csrf_token: "test",
                          ready: true,
                          version: "0.0.1",
                      }
                    : url.includes("/runs")
                      ? { items: [], total: 0, offset: 0, limit: 50 }
                      : url.endsWith("/sources")
                        ? []
                        : {},
        })),
    );
    render(
        <MemoryRouter initialEntries={[path]}>
            <App />
        </MemoryRouter>,
    );
}
test("B8: пустой список прогонов объясняет подключение источника", async () => {
    mount("/runs");
    expect(await screen.findByText(/Прогонов пока нет/)).toBeTruthy();
    expect(
        screen.getByRole("link", { name: "Подключить источник" }),
    ).toBeTruthy();
});
test("B1: навигация доступна с прямой ссылки", async () => {
    mount("/findings");
    expect(
        await screen.findByRole("navigation", { name: "Разделы проекта" }),
    ).toBeTruthy();
});
test("B6: форма содержит основную метрику задачи", async () => {
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => ({
            ok: true,
            json: async () =>
                url.endsWith("/workspace")
                    ? {
                          root: "/project",
                          csrf_token: "test",
                          ready: true,
                          version: "0.0.1",
                      }
                    : {
                          text: "task: {type: classification/binary, output: probabilities}\n",
                          etag: "v1",
                          path: "/project/task.yaml",
                          valid: true,
                          exists: true,
                      },
        })),
    );
    render(
        <MemoryRouter initialEntries={["/task"]}>
            <App />
        </MemoryRouter>,
    );
    expect(await screen.findByLabelText("Основная метрика")).toBeTruthy();
});
