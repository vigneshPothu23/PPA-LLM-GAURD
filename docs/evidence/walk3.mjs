import { chromium } from 'playwright';
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto('http://localhost:8596'); await p.waitForTimeout(5000);
const t = await p.locator('[data-testid="stMain"]').first().innerText();
console.log((t.includes('Sign-in is not configured') && !t.includes('Command Centre') ? 'PASS' : 'FAIL') + ' no auth configured -> no data shown (re-run on current code)');
await p.screenshot({ path: '/home/claude/ppa-guard/docs/evidence/screenshots/10_unconfigured_fail_closed.png' });
await b.close();
