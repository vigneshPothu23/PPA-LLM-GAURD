// Fail-closed and sign-in checks against three running instances. Usage: node walk2.mjs <outdir>
import { chromium } from 'playwright';
const [,, OUT] = process.argv;
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const log = [];
const check = (c, m) => log.push((c ? 'PASS ' : 'FAIL ') + m);
const open = async (url) => {
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await p.goto(url); await p.waitForTimeout(4000);
  await p.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 60000 }).catch(() => {});
  return p;
};
const main = (p) => p.locator('[data-testid="stMain"]').first().innerText();

// 1. no sign-in configured
let p = await open('http://localhost:8596');
let t = await main(p);
check(t.includes('Sign-in is not configured') && !t.includes('Command Centre'), 'no auth configured -> no data shown');
await p.screenshot({ path: `${OUT}/10_unconfigured_fail_closed.png` });

// 2. OIDC configured (dummy credentials): sign-in page only, nothing else reachable
p = await open('http://localhost:8597');
t = await main(p);
check(t.includes('Continue with Google') && t.includes('Continue with Microsoft') && !t.includes('Scan Console'), 'OIDC configured -> sign-in page only');
await p.screenshot({ path: `${OUT}/11_oidc_sign_in.png` });
for (const path of ['/scan', '/members', '/audit']) {
  const q = await open('http://localhost:8597' + path);
  const tt = await main(q);
  check(tt.includes('Sign in') && !tt.includes('Untrusted input') && !tt.includes('Invite a member'), `unauthenticated ${path} -> sign-in page`);
  await q.close();
}
const [req] = await Promise.all([
  p.waitForRequest(r => r.url().includes('accounts.google.com') || r.url().includes('/auth/login'), { timeout: 15000 }).catch(() => null),
  p.getByRole('button', { name: 'Continue with Google' }).click(),
]);
await p.waitForTimeout(3000);
check(req !== null || p.url().includes('google') || p.url().includes('/auth/login'), `Google button starts OIDC redirect (${req ? req.url().slice(0, 60) : p.url().slice(0, 60)})`);

// 3. model missing: every scan blocks
p = await open('http://localhost:8598');
const sb = p.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Demo persona' });
await sb.click(); await p.keyboard.type('member@acme.example'); await p.keyboard.press('Enter'); await p.waitForTimeout(2500);
await p.locator('[data-testid="stSidebarNav"] a').filter({ hasText: 'Scan Console' }).click(); await p.waitForTimeout(2500);
await p.getByRole('button', { name: 'Benign question' }).click(); await p.waitForTimeout(1500);
await p.getByRole('button', { name: 'Scan', exact: true }).click(); await p.waitForTimeout(3000);
t = await main(p);
check(t.includes('BLOCK') && t.includes('fail closed') && !t.includes('Protected prompt'), 'Engine 2 missing -> benign input BLOCKED, no prompt');
await p.screenshot({ path: `${OUT}/12_model_missing_fail_closed.png` });
console.log(log.join('\n'));
await b.close();
