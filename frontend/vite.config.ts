import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    build: { outDir: "../src/mlreview/web/static", emptyOutDir: true },
    test: { environment: "jsdom", include: ["src/**/*.test.tsx"] },
});
