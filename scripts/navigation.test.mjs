import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../static/navigation.js', import.meta.url), 'utf8');
function browser() {
  const events = [], visits = [];
  const location = {origin:'https://demo.vercel.app', pathname:'/admin', search:''};
  const history = {pushState(state, title, path) {
    visits.push(path); const url = new URL(path, location.origin);
    location.pathname = url.pathname; location.search = url.search;
  }};
  const window = {dispatchEvent: event => events.push(event), open() { throw new Error('Must not open a tab'); }};
  class CustomEvent { constructor(type, options) {this.type = type; this.detail = options.detail;} }
  vm.runInNewContext(source, {window, location, history, URL, CustomEvent});
  return {nav:window.AccidenNavigation, events, visits};
}
test('all three dashboard buttons switch views in the current tab', () => {
  const b = browser();
  for (const route of ['/ambulance', '/police', '/admin', '/police', '/ambulance']) b.nav.navigate(route);
  assert.deepEqual(b.visits, ['/ambulance', '/police', '/admin', '/police', '/ambulance']);
  assert.equal(b.events.at(-1).detail.view, 'ambulance');
});
test('clicking the current dashboard does not add duplicate history', () => {
  const b = browser(); b.nav.navigate('/admin'); b.nav.navigate('/police'); b.nav.navigate('/police');
  assert.deepEqual(b.visits, ['/police']);
});
test('review selects the incident in the same tab', () => {
  const b = browser(); b.nav.navigate('/ambulance?incident=INC-test');
  assert.equal(b.events[0].detail.incident, 'INC-test');
  assert.equal(b.visits[0], '/ambulance?incident=INC-test');
  assert.throws(() => b.nav.navigate('https://outside.example/admin'));
});
test('password controls and popup navigation are absent', () => {
  const html = readFileSync(new URL('../static/index.html', import.meta.url), 'utf8');
  const app = readFileSync(new URL('../static/app.js', import.meta.url), 'utf8');
  assert.doesNotMatch(html, /workspacePassword|workspaceLogin|target="_blank"/);
  assert.doesNotMatch(app, /window\.open|acciden-password|Authorization:/);
});
