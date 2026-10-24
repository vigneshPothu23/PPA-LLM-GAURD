// Browser walkthrough of the running PPA Guard console (demo mode). Usage: node walk.mjs <url> <outdir>
import { chromium } from 'playwright';
const [,, URL, OUT] = process.argv;
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
const log = [];
const settle = async () => {
  await p.waitForTimeout(400);
  await p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 60000 }).catch(() => {});
  await p.waitForTimeout(600);
};
const shot = async (name) => { await p.screenshot({ path: `${OUT}/${name}.png`, fullPage: true }); log.push(`shot ${name}`); };
const persona = async (email) => {
  const sb = p.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Demo persona' });
  await sb.click(); await p.keyboard.type(email); await p.keyboard.press('Enter'); await settle();
};
const nav = async (title) => { await p.locator('[data-testid="stSidebarNav"] a').filter({ hasText: title }).first().click(); await settle(); };
const navNames = async () => (await p.locator('[data-testid="stSidebarNav"] a').allInnerTexts()).map(s => s.trim().split('\n').pop().trim());
const body = async () => (await p.locator('section.stMain, [data-testid="stMain"]').first().innerText());
const pickApp = async (name) => {
  const sb = p.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Application' }).first();
  await sb.click(); await p.keyboard.type(name); await p.keyboard.press('Enter'); await settle();
};
const scan = async (text) => {
  const ta = p.locator('textarea[aria-label="Untrusted input"]');
  await ta.fill(text); await ta.press('Control+Enter'); await settle();
  await p.getByRole('button', { name: 'Scan', exact: true }).click(); await settle();
};
const check = (cond, msg) => { log.push((cond ? 'PASS ' : 'FAIL ') + msg); };

await p.goto(URL); await p.getByRole('heading', { name: 'Command Centre' }).waitFor({ timeout: 60000 }).catch(() => {}); await settle();
check((await body()).includes('Command Centre'), 'app loads (default persona = first seeded user)');

for (const [email, expected] of [
  ['member@acme.example', ['Command Centre', 'Scan Console', 'Scan History', 'System & Integrations']],
  ['auditor@acme.example', ['Command Centre', 'Scan History', 'Applications & Policy', 'Members', 'Audit Log', 'System & Integrations']],
  ['admin@acme.example', ['Command Centre', 'Scan Console', 'Review Queue', 'Scan History', 'Applications & Policy', 'Members', 'Audit Log', 'System & Integrations']]]) {
  await persona(email);
  const got = await navNames();
  check(JSON.stringify(got) === JSON.stringify(expected), `${email} sidebar = ${got.join(' | ')}`);
}

await persona('member@acme.example');
await nav('Scan Console');
await p.getByRole('button', { name: 'Direct injection' }).click(); await settle();
await p.getByRole('button', { name: 'Scan', exact: true }).click(); await settle();
let t = await body();
check(/ALLOW|WARN|BLOCK/.test(t) && t.includes('record '), 'member scan returns a recorded decision');
await shot('01_member_scan_stub_model');

await pickApp('mail-assistant');
await scan('Please summarise this note.​');
t = await body();
check(t.includes('Held for a Security Analyst'), 'hidden character on general profile -> WARN held');
await shot('02_member_warn_held');

// direct URL to an admin page in a new tab: a fresh session (default persona) is fine; the point is members-only pages
await persona('analyst@acme.example');
await nav('Review Queue');
await shot('03_analyst_review_queue');
const before = await p.getByRole('button', { name: 'Release once' }).count();
await p.getByRole('button', { name: 'Release once' }).last().click(); await settle();
check((await p.getByRole('button', { name: 'Release once' }).count()) === before - 1, `analyst released one held input (${before} open before)`);

await persona('member@acme.example');
await nav('Scan Console');
await pickApp('mail-assistant');
await scan('Please summarise this note.​');
t = await body();
check(t.includes('analyst release used') && t.includes('Protected prompt'), 'released input forwarded once with PPA prompt');
await shot('04_member_release_used_ppa_prompt');
await p.getByRole('button', { name: 'Scan', exact: true }).click(); await settle();
check((await body()).includes('Held for a Security Analyst'), 'second resubmission held again (single use)');

await persona('auditor@acme.example');
await nav('Audit Log');
await p.getByRole('button', { name: 'Verify chain' }).click(); await settle();
check((await body()).includes('CHAIN INTACT'), 'auditor verifies audit chain');
await shot('05_auditor_audit_log');

await persona('admin@acme.example');
await nav('Command Centre'); await shot('06_admin_command_centre');
await nav('Applications & Policy'); await shot('07_admin_applications_policy');
await nav('Members'); await shot('08_admin_members');
await nav('System & Integrations'); await shot('09_system_integrations');

// a member typing an admin URL gets the default page, not the admin page
const p2 = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p2.goto(URL); await p2.waitForTimeout(2500);
const sb = p2.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Demo persona' });
await sb.click(); await p2.keyboard.type('member@acme.example'); await p2.keyboard.press('Enter'); await p2.waitForTimeout(2500);
await p2.evaluate(() => window.history.pushState({}, '', '/members'));
await p2.locator('[data-testid="stSidebarNav"] a').first().click(); await p2.waitForTimeout(2000);
check(!(await p2.locator('[data-testid="stMain"]').first().innerText()).includes('Invite a member'), 'member never sees member-management UI');
console.log(log.join('\n'));
await b.close();
