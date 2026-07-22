import { defineConfig, devices } from "@playwright/test";

/**
 * E2E configuration.
 *
 * The API is NOT started here: it loads a model and binds a GPU, which is too
 * heavy and too environment-specific for a test runner to own. Start it
 * separately (see docs/INFERENCE_GUIDE.md); the specs skip themselves with a
 * clear message when it is not reachable.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3100",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3100",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
