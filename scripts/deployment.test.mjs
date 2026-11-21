import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = fileURLToPath(new URL('../', import.meta.url));
function build(origin) {
  return spawnSync(process.execPath, ['scripts/build.mjs'], {
    cwd: root, encoding: 'utf8',
    env: { ...process.env, VERCEL: '1', API_BASE_URL: origin },
  });
}
test('Vercel build produces public assets and backend configuration only', () => {
  const result = build('https://api.example.com/');
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(readdirSync(new URL('../dist/', import.meta.url)).sort(), ['index.html', 'static']);
  assert.deepEqual(readdirSync(new URL('../dist/static/', import.meta.url)).sort(), ['app.css', 'app.js', 'config.js', 'navigation.js']);
  const context = { window: {} };
  vm.runInNewContext(readFileSync(new URL('../dist/static/config.js', import.meta.url), 'utf8'), context);
  assert.equal(context.window.ACCIDEN_CONFIG.apiBaseUrl, 'https://api.example.com');
});
test('Vercel build rejects insecure and malformed external backend origins', () => {
  for (const origin of ['http://api.example.com', 'https://api.example.com/api', 'https://user:secret@api.example.com', 'not-a-url']) {
    assert.notEqual(build(origin).status, 0, origin);
  }
});
test('API and video URLs use the backend while local mode keeps relative paths', () => {
  const source = readFileSync(new URL('../static/app.js', import.meta.url), 'utf8').split('const $ =')[0];
  for (const origin of ['', 'https://api.example.com/']) {
    const context = { window: { ACCIDEN_CONFIG: { apiBaseUrl: origin } }, URL };
    vm.runInNewContext(source, context);
    for (const path of ['/api/incidents', '/static/clips/clip_INC-TEST.mp4']) {
      const actual = vm.runInNewContext(`backendUrl(${JSON.stringify(path)})`, context);
      assert.equal(actual, origin ? `https://api.example.com${path}` : path);
    }
  }
});
test('every dashboard URL has an explicit Vercel rewrite', () => {
  const config = JSON.parse(readFileSync(new URL('../vercel.json', import.meta.url), 'utf8'));
  for (const route of ['/admin', '/vision', '/ambulance', '/police']) {
    assert.ok(config.rewrites.some(rule => rule.source === route && rule.destination === '/index.html'));
  }
});

test('same-origin build works without a separate backend host', () => { const result = build(''); assert.equal(result.status, 0, result.stderr); });

test('storage controls are restricted to Admin dashboard', () => {
  const source = readFileSync(new URL('../static/app.js', import.meta.url), 'utf8');
  const body = source.split('function renderStorage() {')[1].split('async function deleteVideo(')[0];
  const elements = { storagePanel: {}, sharingPanel: {}, manageVideos: {}, storageSummary: {}, storageRows: {} };
  for (const view of ['vision', 'ambulance', 'police']) {
    vm.runInNewContext('function renderStorage() {' + body + '; renderStorage();', {
      state: { view, incidents: [] }, $: id => elements[id], escapeHTML: String
    });
    assert.equal(elements.storagePanel.hidden, view !== 'vision');
  }
});

test('Admin incident rows expose delete next to the incident ID', () => {
  const source = readFileSync(new URL('../static/app.js', import.meta.url), 'utf8');
  const render = source.split('function render() {')[1].split("$('incidentRows').onclick")[0];
  const elements = new Proxy({}, { get: (target, id) => target[id] ||= { value: id === 'filter' ? 'all' : '', querySelector: () => ({}) } });
  const context = { state: { view: 'vision', selected: null, incidents: [{id:'INC-test', has_clip:true, timestamp:'now', location:'Test'}] }, $: id => elements[id], responseStatus: () => 'PENDING', pill: () => '', escapeHTML: String };
  vm.runInNewContext('function render() {' + render + ';render();', context);
  assert.match(elements.incidentRows.innerHTML, /data-delete-video="INC-test"/);
  context.state.view = 'police';
  vm.runInNewContext('render();', context);
  assert.doesNotMatch(elements.incidentRows.innerHTML, /data-delete-video/);
});
