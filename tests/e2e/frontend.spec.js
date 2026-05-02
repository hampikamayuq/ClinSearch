const { test, expect } = require('@playwright/test');

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem('cs_v3', JSON.stringify({
      userKey: 'test-key',
      userProvider: 'openai',
      workspace: [],
      notes: [],
      reviewDepth: 'fast'
    }));
  });
  await page.route('https://clinsearch.onrender.com/health', async route => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        ai: {
          gemini: { status: 'ok', configured: true, last_latency_ms: 120 },
          groq: { status: 'configured', configured: true }
        },
        cache: { persistent_search_entries: 3, persistent_tool_entries: 2 }
      })
    });
  });
  await page.route('https://clinsearch.onrender.com/api/quota', async route => {
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ unlimited: true }) });
  });
  await page.route('https://clinsearch.onrender.com/api/session', async route => {
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ id: 'test-session', sessions: [] }) });
  });
  await page.route('https://clinsearch.onrender.com/metrics', async route => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        cache: { persistent_search_entries: 3, persistent_tool_entries: 2 },
        database: { ok: true, path: '/var/data/clinsearch.db', sessions: 1, workspace_items: 0, alerts: 0 },
        providers: {
          gemini: { status: 'ok', configured: true, calls: 2, errors: 0, rate_limited: 0, last_latency_ms: 120 },
          groq: { status: 'configured', configured: true, calls: 0, errors: 0, rate_limited: 0 }
        },
        endpoints: { '/api/search': { count: 4, errors: 0, last_latency_ms: 88, max_latency_ms: 120 } }
      })
    });
  });
});

test('core navigation and clinical controls render', async ({ page }) => {
  await page.goto('/index.html');

  await expect(page.locator('#mt-search')).toContainText('Search Evidence');
  await expect(page.locator('#providerStrip')).toContainText('gemini: ok');
  await expect(page.locator('#picoPatient')).toBeVisible();

  await page.getByRole('button', { name: 'Treatment' }).click();
  await expect(page.locator('#picoChip')).toHaveClass(/on/);
  await expect(page.locator('#inp')).toHaveValue(/should \[intervention\]/);

  await page.getByText('Calculators').first().click();
  await expect(page.getByText('Bedside calculators and risk tools')).toBeVisible();
  await page.getByRole('button', { name: 'Risk scores' }).click();
  await expect(page.getByText('CHA₂DS₂-VASc Score')).toBeVisible();

  await page.locator('#mt-system').click();
  await expect(page.getByText('System Monitoring')).toBeVisible();
  await expect(page.getByText('/var/data/clinsearch.db')).toBeVisible();
});

test('mobile drawer, workspace, and tables stay reachable', async ({ page }) => {
  await page.goto('/index.html');
  await page.setViewportSize({ width: 390, height: 844 });

  await page.getByTitle('Menu').click();
  await expect(page.locator('.lsb')).toHaveClass(/open/);
  await page.locator('#lsbOverlay').click();
  await expect(page.locator('.lsb')).not.toHaveClass(/open/);

  await page.locator('.mode-tab', { hasText: 'Saved Papers' }).click();
  await expect(page.locator('#workspace')).toHaveClass(/open/);
  await expect(page.getByText('Save papers here')).toBeVisible();
  await expect(page.locator('#workspace')).toHaveCSS('position', 'fixed');

  await page.evaluate(() => {
    const panel = window.addToolPanel('Evidence Table Test');
    window.showEvidenceTable([{
      title: 'A randomized trial in adults',
      year: '2025',
      study_type: 'RCT',
      abstract: 'Adults treated with drug compared with placebo. Primary outcome mortality improved.',
      url: 'https://example.org',
      doi: '10.1/test'
    }], 'test');
  });
  await expect(page.getByText('Intervention / Comparator').last()).toBeVisible();
});
