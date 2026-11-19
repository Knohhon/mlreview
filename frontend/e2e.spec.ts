import { test, expect } from "@playwright/test";
test("B1, B3, B5, B8: прогон → ревью → находка → решение → reload", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/runs");
    await page.getByRole("link", { name: "candidate candidate" }).click();
    await expect(
        page.getByRole("heading", { name: "candidate", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Проверить контракт" }).click();
    await expect(
        page.getByRole("heading", { name: "Проверка контракта" }),
    ).toBeVisible();
    await page.getByRole("link", { name: "Ревью прогона" }).click();
    await page.getByRole("button", { name: "Запустить ревью" }).click();
    await expect(page.getByRole("heading", { name: /Отчёт/ })).toBeVisible({
        timeout: 30000,
    });
    await expect(
        page.getByRole("heading", { name: /Что не проверено/ }),
    ).toBeVisible();
    const finding = page.locator('table a[href^="/findings/"]').first();
    await finding.click();
    await expect(
        page.getByRole("heading", { name: "Наблюдаемые факты" }),
    ).toBeVisible();
    await page
        .getByRole("textbox", { name: "Причина изменения статуса" })
        .fill("Проверено в браузере");
    await page.getByRole("button", { name: "Подтвердить отклонение" }).click();
    await expect(page.getByText("Причина: Проверено в браузере")).toBeVisible();
    await page.reload();
    await expect(page.getByText("Причина: Проверено в браузере")).toBeVisible();
    await page.getByRole("link", { name: "Добавить решение" }).click();
    await page
        .getByRole("textbox", { name: "Текст записи" })
        .fill("<script>window.pwned=true</script>");
    await page
        .getByRole("button", { name: "Добавить запись", exact: true })
        .click();
    await expect(
        page.getByRole("link", { name: "<script>window.pwned=true</script>" }),
    ).toBeVisible();
    expect(
        await page.evaluate(() => Reflect.get(window, "pwned")),
    ).toBeUndefined();
    await page.screenshot({
        path: "test-results/review-journal.png",
        fullPage: true,
    });
    expect(errors).toEqual([]);
});
test("B6: YAML требует просмотра изменений перед сохранением", async ({
    page,
}) => {
    await page.goto("/task");
    const save = page.getByRole("button", { name: "Подтвердить сохранение" });
    await expect(save).toBeDisabled();
    const editor = page.getByRole("textbox", { name: "YAML задачи" });
    await expect(editor).toContainText("task:");
    await editor.fill((await editor.inputValue()) + "\n# browser change\n");
    await page.getByRole("button", { name: "Сравнить изменения" }).click();
    await expect(save).toBeEnabled();
    await save.click();
    await expect(page.getByText("Сохранено с резервной копией")).toBeVisible();
});
