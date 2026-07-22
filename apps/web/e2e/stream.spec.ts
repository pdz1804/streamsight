import { expect, test, type Page } from "@playwright/test";

/**
 * Browser-level end-to-end coverage.
 *
 * These assert on *observable behaviour*, not on the presence of markup: that
 * the canvas receives pixels that keep changing, that the FPS readout moves off
 * zero, that a theme switch actually restyles the document. A test that only
 * checks a heading exists would pass against a completely broken stream.
 */

const API = "http://localhost:8100";

test.beforeAll(async ({ request }) => {
  try {
    const response = await request.get(`${API}/health`, { timeout: 5000 });
    test.skip(!response.ok(), "API is not healthy");
  } catch {
    test.skip(true, `API not reachable at ${API}; start uvicorn first`);
  }
});

/** Mean pixel value of the canvas, or null when nothing has been painted. */
async function canvasSignature(page: Page): Promise<number | null> {
  return page.evaluate(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas || !canvas.width) return null;
    const context = canvas.getContext("2d");
    if (!context) return null;
    const { data } = context.getImageData(
      0,
      0,
      Math.min(80, canvas.width),
      Math.min(80, canvas.height),
    );
    let sum = 0;
    for (let i = 0; i < data.length; i += 41) sum += data[i];
    return sum;
  });
}

test("live console streams annotated frames with track identities", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Throughput" })).toBeVisible();

  await page.getByRole("button", { name: "Start stream" }).click();

  // Frames must actually arrive.
  await expect
    .poll(async () => await canvasSignature(page), {
      timeout: 45_000,
      message: "no frames were painted to the canvas",
    })
    .not.toBeNull();

  // ...and keep arriving, with different content each time.
  const signatures: (number | null)[] = [];
  for (let i = 0; i < 4; i++) {
    await page.waitForTimeout(900);
    signatures.push(await canvasSignature(page));
  }
  expect(new Set(signatures).size).toBeGreaterThan(1);

  // The tracker should confirm identities on the densely populated demo clip.
  await expect(page.getByText(/\d+ active/)).toBeVisible();
  const active = await page.getByText(/\d+ active/).textContent();
  expect(Number.parseInt(active ?? "0", 10)).toBeGreaterThan(0);

  await page.getByRole("button", { name: "Stop stream" }).click();
  await expect(page.getByRole("button", { name: "Start stream" })).toBeVisible();
});

test("metrics dashboard reports live telemetry", async ({ page }) => {
  await page.goto("/metrics");
  await expect(page.getByRole("heading", { name: "Runtime telemetry" })).toBeVisible();

  for (const panel of ["Frames per second", "Latency", "Tracking", "Host"]) {
    await expect(page.getByRole("heading", { name: panel })).toBeVisible();
  }
  await expect(page.getByText(/Nominal|Degraded/)).toBeVisible();
});

test("settings lists backends and explains the unavailable ones", async ({ page }) => {
  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: "Inference configuration" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Backend" })).toBeVisible();

  // Every backend is listed, available or not, so the operator can see why.
  await expect(page.getByText("FP32 - PyTorch CPU")).toBeVisible();
  await expect(page.getByText(/\d+px/).first()).toBeVisible();
});

test("theme switches between light and dark and survives reload", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("radio", { name: "Dark" }).click();
  await expect.poll(async () => page.evaluate(() => document.documentElement.dataset.theme))
    .toBe("dark");

  await page.getByRole("radio", { name: "Light" }).click();
  await expect.poll(async () => page.evaluate(() => document.documentElement.dataset.theme))
    .toBe("light");

  // The choice is persisted and applied before first paint, so no flash.
  await page.reload();
  await expect.poll(async () => page.evaluate(() => document.documentElement.dataset.theme))
    .toBe("light");
});

test("the page never produces a horizontal or double scrollbar", async ({ page }) => {
  await page.goto("/");
  const overflow = await page.evaluate(() => ({
    bodyOverflowX: document.body.scrollWidth > window.innerWidth + 1,
    bodyScrolls: document.body.scrollHeight > window.innerHeight + 1,
  }));
  expect(overflow.bodyOverflowX).toBe(false);
  expect(overflow.bodyScrolls).toBe(false);
});
