#!/usr/bin/env node
//
// Captures anonymized screenshots of the Aquila dashboard for documentation.
// Requires a running frontend (default: http://localhost:5173) backed by a
// live host backend with real data.
//
// Usage:
//   npm install          # first time only
//   node update-screenshots.mjs [--url http://localhost:5173] [--out ../assets/img]
//
import { chromium } from "playwright";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { parseArgs } from "util";

const __dirname = dirname(fileURLToPath(import.meta.url));

const { values: flags } = parseArgs({
  options: {
    url: { type: "string", default: "http://localhost:5173" },
    out: { type: "string", default: resolve(__dirname, "../assets/img") },
  },
  strict: false,
});

const BASE_URL = flags.url;
const OUT_DIR = resolve(flags.out);

// ---------------------------------------------------------------------------
// Anonymization
// ---------------------------------------------------------------------------

const HOSTNAME_POOL = ["atlas", "boreas", "calypso", "delphi", "europa"];
const IP_POOL = [
  "10.0.1.10",
  "10.0.1.11",
  "10.0.1.12",
  "10.0.1.13",
  "10.0.1.14",
];
const OWNER_POOL = ["alice", "bob", "carol", "dave", "eve", "frank"];
const PORT_OFFSET = -3000;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function buildAnonMap() {
  const [nodes, deployments, apiKeys] = await Promise.all([
    fetchJson(`${BASE_URL}/api/nodes/`),
    fetchJson(`${BASE_URL}/api/deployments/`),
    fetchJson(`${BASE_URL}/api/api-keys`).catch(() => []),
  ]);

  const map = new Map();

  const hostnames = [...new Set(nodes.map((n) => n.hostname))];
  const ips = [...new Set(nodes.map((n) => n.ip_address))];
  const owners = [
    ...new Set([
      ...deployments.map((d) => d.owner).filter(Boolean),
      ...apiKeys.map((k) => k.label).filter(Boolean),
    ]),
  ];
  const ports = [
    ...new Set([
      ...nodes.map((n) => n.port).filter(Boolean),
      ...deployments.map((d) => d.port),
    ]),
  ];

  hostnames.forEach((h, i) => map.set(h, HOSTNAME_POOL[i % HOSTNAME_POOL.length]));
  ips.forEach((ip, i) => map.set(ip, IP_POOL[i % IP_POOL.length]));
  owners.forEach((o, i) => map.set(o, OWNER_POOL[i % OWNER_POOL.length]));
  ports.forEach((p) => map.set(String(p), String(p + PORT_OFFSET)));
  apiKeys.forEach((k, i) => {
    if (k.prefix) map.set(k.prefix, `vcm-${String(i + 1).padStart(4, "0")}`);
  });

  return map;
}

function makeAnonymizer(map) {
  const entries = [...map.entries()].sort((a, b) => b[0].length - a[0].length);
  return (text) => {
    let out = text;
    for (const [real, fake] of entries) {
      out = out.replaceAll(real, fake);
    }
    return out;
  };
}

// ---------------------------------------------------------------------------
// Screenshot helpers
// ---------------------------------------------------------------------------

async function screenshotElement(locator, filename) {
  const el = locator.first();
  await el.waitFor({ state: "visible", timeout: 5000 });
  await el.screenshot({ path: resolve(OUT_DIR, filename) });
  console.log(`  ✓ ${filename}`);
}

async function openDialogAndScreenshot(page, clickLocator, filename) {
  await clickLocator.first().click();
  const dialog = page.locator(".MuiDialog-paper").first();
  await dialog.waitFor({ state: "visible", timeout: 5000 });
  await page.waitForTimeout(400);
  await dialog.screenshot({ path: resolve(OUT_DIR, filename) });
  console.log(`  ✓ ${filename}`);
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log(`Discovering data from ${BASE_URL} ...`);
  const anonMap = await buildAnonMap();
  const anonymize = makeAnonymizer(anonMap);

  console.log(
    `Anonymizing ${anonMap.size} values:`,
    Object.fromEntries(anonMap)
  );

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();

  // Intercept all API responses and anonymize them.
  await page.route("**/api/**", async (route) => {
    try {
      const response = await route.fetch();
      const contentType = response.headers()["content-type"] || "";
      if (contentType.includes("json") || contentType.includes("text")) {
        const body = await response.text();
        await route.fulfill({
          status: response.status(),
          headers: response.headers(),
          body: anonymize(body),
        });
      } else {
        await route.fulfill({ response });
      }
    } catch {
      await route.continue();
    }
  });

  // Block WebSocket to prevent un-anonymized live updates.
  await page.route("**/ws", (route) => route.abort());
  await page.route("**/ws/**", (route) => route.abort());

  console.log("Loading dashboard ...");
  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);

  // Fix the "Live" indicator to show green (WebSocket is blocked).
  await page.evaluate(() => {
    const liveSpans = [...document.querySelectorAll("span")].filter(
      (el) => el.textContent?.trim() === "Live"
    );
    for (const span of liveSpans) {
      const dot = span.parentElement?.querySelector("span");
      if (dot && dot !== span) {
        dot.style.color = "#22c55e";
      }
      const circle = span.parentElement?.querySelector("svg circle, [class*='dot']");
      if (circle) circle.style.fill = "#22c55e";
    }
  });

  console.log("Taking screenshots ...");

  // 1. Full dashboard hero shot — wide viewport for a panoramic overview.
  await page.setViewportSize({ width: 2000, height: 1050 });
  await page.waitForTimeout(400);
  await page.screenshot({
    path: resolve(OUT_DIR, "aquila-screenshot.png"),
    fullPage: false,
  });
  console.log("  ✓ aquila-screenshot.png");

  // Restore default viewport for element screenshots.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(300);

  // 2. Deploy form panel
  await screenshotElement(page.locator(".deploy-rail"), "deploy-form.png");

  // Widen viewport so tables render without horizontal scroll clipping.
  await page.setViewportSize({ width: 2200, height: 900 });
  await page.waitForTimeout(500);

  // 3. Model Deployments table panel
  const stageChildren = page.locator(".stage > .panel");
  await screenshotElement(stageChildren.first(), "deployment-table.png");

  // 4. Deployment settings dialog — click first deployment row's settings icon
  const depSettingsBtn = stageChildren
    .first()
    .locator('[aria-label="Settings"]')
    .first();
  const hasDeployments = (await depSettingsBtn.count()) > 0;
  if (hasDeployments) {
    await openDialogAndScreenshot(
      page,
      depSettingsBtn,
      "deployment-settings-dialog.png"
    );
  } else {
    console.log("  ⚠ No deployments — skipping deployment-settings-dialog.png");
  }

  // 5. Nodes table panel (second panel in .stage)
  await screenshotElement(stageChildren.nth(1), "node-table.png");

  // 6. Node metrics — click metrics button on first node, screenshot expanded panel
  const metricsBtn = page.locator('[aria-label="Metrics"]').first();
  const hasNodes = (await metricsBtn.count()) > 0;
  if (hasNodes) {
    await metricsBtn.click();
    await page.waitForTimeout(1500);
    await screenshotElement(stageChildren.nth(1), "node-metrics.png");
    await metricsBtn.click();
    await page.waitForTimeout(300);
  } else {
    console.log("  ⚠ No nodes — skipping node-metrics.png");
  }

  // Restore normal viewport for dialogs.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(300);

  // 7. Node manage dialog
  const manageBtn = page.locator('[aria-label="Manage"]').first();
  if (hasNodes) {
    await openDialogAndScreenshot(page, manageBtn, "node-manage-dialog.png");
  } else {
    console.log("  ⚠ No nodes — skipping node-manage-dialog.png");
  }

  // 8. Settings dialog — click the gear icon in the header
  const headerSettingsBtn = page.locator(".brand [aria-label='Settings']");
  await openDialogAndScreenshot(
    page,
    headerSettingsBtn,
    "settings-dialog.png"
  );

  await browser.close();
  console.log(`\nDone! ${8} screenshots saved to ${OUT_DIR}`);
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
