// Records a real walkthrough of the running console (no mocked output) and saves screenshots.
//   PPAGUARD_DEMO_MODE=1 streamlit run streamlit_app.py --server.port 8599 --server.headless true
//   node scripts/demo/record_demo.mjs http://localhost:8599 <out_dir>
// Produces <out_dir>/video/*.webm (convert with ffmpeg) and <out_dir>/NN_*.png. Every verdict shown is whatever the
// application returns; the script only asserts and logs it (outcomes.json).
import { chromium } from 'playwright';
import fs from 'fs';

const URL = process.argv[2] || 'http://localhost:8599';
const OUT = process.argv[3] || 'demo_out';
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, recordVideo: { dir: `${OUT}/video`, size: { width: 1440, height: 900 } } });
const p = await ctx.newPage();
const outcomes = [];
let shot = 0;

async function caption(text, ms = 3500) {
  await p.evaluate((t) => {
    let d = document.getElementById('demo-caption');
    if (!d) {
      d = document.createElement('div'); d.id = 'demo-caption';
      Object.assign(d.style, { position: 'fixed', left: '50%', bottom: '18px', transform: 'translateX(-50%)', zIndex: 999999,
        background: 'rgba(20,28,32,.92)', color: '#fff', padding: '10px 18px', borderRadius: '8px', font: '17px Arial',
        maxWidth: '1100px', textAlign: 'center', boxShadow: '0 4px 16px rgba(0,0,0,.3)' });
      document.body.appendChild(d);
    }
    d.textContent = t;
  }, text);
  await p.waitForTimeout(ms);
}
const main = () => p.locator('[data-testid="stMain"]');
async function idle() {
  await p.waitForTimeout(600);
  await p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 180000 }).catch(() => {});
  await p.waitForTimeout(800);
}
async function snap(name) { shot += 1; await p.screenshot({ path: `${OUT}/${String(shot).padStart(2, '0')}_${name}.png`, fullPage: false }); }
async function persona(email) {
  await p.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first().click();
  await p.locator('[role="option"]', { hasText: email }).first().click();
  await idle();
}
async function page(title) { await p.locator('[data-testid="stSidebarNav"] a', { hasText: title }).first().click(); await idle(); }
async function selectApp(name) {
  await main().locator('[data-testid="stSelectbox"]').first().click();
  await p.locator('[role="option"]', { hasText: name }).first().click();
  await idle();
}
async function scanExample(label, expect, note) {
  await main().getByRole('button', { name: label, exact: true }).click();
  await idle();
  await main().getByRole('button', { name: 'Scan', exact: true }).click();
  await p.waitForTimeout(1000);
  await main().getByText('Why', { exact: true }).waitFor({ timeout: 180000 });
  await idle();
  const text = await main().innerText();
  const decision = ['BLOCK', 'WARN', 'ALLOW'].find((d) => new RegExp(`\\b${d}\\b`).test(text.split('Why')[0].split('Untrusted input').pop()));
  outcomes.push({ example: label, expected: expect, observed: decision, pass: decision === expect });
  console.log(`${decision === expect ? 'PASS' : 'MISMATCH'} ${label}: observed ${decision}, expected ${expect}`);
  await main().getByText('Why', { exact: true }).scrollIntoViewIfNeeded();
  await p.mouse.wheel(0, -420);
  await caption(`${label}: ${decision}. ${note}`, 5500);
  await snap(label.toLowerCase().replace(/[^a-z0-9]+/g, '_'));
}

await p.goto(URL);
await p.getByText('Demo persona').first().waitFor({ timeout: 300000 });
await idle();
await caption('PPA Guard: screens untrusted text before an LLM application uses it, and decides ALLOW, WARN or BLOCK. Demo personas only (not authentication).', 5000);
await persona('member@acme.example');
await page('Scan Console');
await selectApp('docs-assistant');
await caption('Ravi (Member) scans documents for the docs-assistant application. Thresholds come from the model card.', 4000);
await scanExample('README with hidden instruction', 'WARN', 'The whole-document score is low (the instruction is diluted), but the context signal finds the hidden comment, so it is held for review.');
await scanExample('Benign SQL code review', 'ALLOW', 'Ordinary review text is forwarded, wrapped by Engine 1 in a fresh random boundary.');
await main().getByText('Protected prompt (Engine 1)').scrollIntoViewIfNeeded().catch(() => {});
await caption('Only an ALLOW returns a protected prompt. PPA Guard never calls an LLM itself.', 4000);
await snap('protected_prompt');
await scanExample('Direct injection', 'BLOCK', 'A direct override scores above 0.99 on the whole document: blocked, nothing forwarded.');
await scanExample('Boundary forgery', 'BLOCK', 'A forged chat boundary is blocked.');
await persona('analyst@acme.example');
await page('Review Queue');
await caption('Ananya (Security Analyst) sees the held README in the Review Queue. A release is single-use and re-checked in full; a BLOCK can never be released.', 6000);
await snap('analyst_review_queue');
await persona('admin@acme.example');
await page('Engine 2 Model');
await caption('Engine 2 Model page: the detector, its pinned source, the context signal and its thresholds. Marked experimental: not validated for production.', 6000);
await snap('model_page');
await p.mouse.wheel(0, 700); await p.waitForTimeout(800);
await caption('Measured results are read from the model card, not typed in. Final holdout results are in docs/v5/FINAL_EVALUATION.md.', 5000);
await snap('model_page_metrics');
await page('System & Integrations');
await caption('System page: what is running and what is not. No downstream LLM is connected; integrations are the console and the Python SDK.', 6000);
await snap('system_page');
await page('Audit Log');
await caption('Every decision is recorded in a hash-chained audit log (hash of the input, never the raw text).', 5000);
await snap('audit_log');
fs.writeFileSync(`${OUT}/outcomes.json`, JSON.stringify(outcomes, null, 2));
await ctx.close();
await browser.close();
console.log('done');
