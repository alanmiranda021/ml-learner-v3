/* node tests/naval_browser.cjs <playwright-core path> [base URL] */
const assert = require('node:assert/strict');
const { chromium } = require(process.argv[2] || 'playwright-core');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage();
    page.on('pageerror', e => console.error('Browser:', e.message));
    await page.goto((process.argv[3] || 'http://127.0.0.1:8080') + '/github-pages/');
    await page.locator('[data-testid="stFileUploader"]').waitFor({ timeout: 240000 });
    const header = 'ship_type,route_id,distance,fuel_type,weather_conditions,engine_efficiency,fuel_consumption,CO2_emissions';
    const rows = Array.from({length: 80}, (_, i) => `Type${i<30?0:1},Route${i%4},${10+i},Diesel,Sun,${70+i/10},${100+i*10+Math.sin(i)*15},${270+i*27}`);
    await page.locator('input[type=file]').setInputFiles({name: 'naval_artificial.csv', mimeType: 'text/csv', buffer: Buffer.from([header, ...rows].join('\n'))});
    await page.getByRole('tab', {name: 'Laboratório — Artigo 3 (naval)'}).click({timeout: 120000});
    await page.getByText('Protocolo do Artigo 3 adaptado', {exact: false}).waitFor();
    const run = page.getByRole('button', {name: 'Executar protocolo naval e gerar PDF'});
    await run.waitFor();
    assert(await run.isEnabled());
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    console.log('Stlite: upload artificial, aba naval e controles disponíveis; nenhuma execução científica disparada.');
    if (process.argv.includes('--run')) {
      async function removeLast(label, count) {
        const widget = page.locator('[data-testid="stMultiSelect"]').filter({has: page.getByText(label, {exact: true})});
        await widget.waitFor();
        const input = widget.locator('input');
        for (let i = 0; i < count; i++) {
          await input.focus();
          await input.press('Backspace');
          await page.waitForTimeout(500);
        }
        await input.press('Escape');
      }
      await removeLast('Grade de neurônios SAE', 3);
      await removeLast('Grade gamma KPCA', 3);
      await removeLast('Grade de componentes KPCA', 2);
      await removeLast('Modelos navais', 1);
      const started = Date.now();
      await run.click();
      const download = page.getByRole('button', {name: 'Baixar relatório naval (.pdf)'});
      await download.waitFor({timeout: 240000});
      assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
      const transfer = page.waitForEvent('download', {timeout: 30000});
      await download.click();
      const artifact = await transfer;
      assert.equal(await artifact.failure(), null);
      const stream = await artifact.createReadStream();
      const chunks = [];
      for await (const chunk of stream) chunks.push(chunk);
      assert(Buffer.concat(chunks).subarray(0, 4).equals(Buffer.from('%PDF')));
      console.log(`Stlite execução artificial SVR: ${((Date.now()-started)/1000).toFixed(1)} s; PDF disponível.`);
    }
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exit(1);});
