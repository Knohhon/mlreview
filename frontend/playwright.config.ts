import { defineConfig } from "@playwright/test";
export default defineConfig({
    testDir: ".",
    testMatch: "e2e.spec.ts",
    workers: 1,
    use: { baseURL: "http://127.0.0.1:8769", headless: true },
    webServer: {
        command: "PYTHONPATH=.. python e2e_workspace.py",
        url: "http://127.0.0.1:8769/api/v1/health",
        timeout: 60000,
        reuseExistingServer: false,
    },
    reporter: "list",
});
