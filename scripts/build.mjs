import { mkdir, copyFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.join(root, 'dist');
const apiBaseUrl = (process.env.API_BASE_URL || '').trim().replace(/\/+$/, '');
if (apiBaseUrl) {
  const parsed = new URL(apiBaseUrl);
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.pathname !== '/' || parsed.search || parsed.hash || parsed.username || parsed.password) {
    throw new Error('API_BASE_URL must be an HTTP(S) origin without a path, query, or credentials.');
  }
  if (process.env.VERCEL && parsed.protocol !== 'https:') throw new Error('Vercel requires an HTTPS backend.');
}
await mkdir(path.join(output, 'static'), { recursive: true });
await copyFile(path.join(root, 'static/index.html'), path.join(output, 'index.html'));
for (const file of ['app.css', 'app.js', 'navigation.js']) {
  await copyFile(path.join(root, 'static', file), path.join(output, 'static', file));
}
await writeFile(path.join(output, 'static/config.js'), `window.ACCIDEN_CONFIG = ${JSON.stringify({ apiBaseUrl })};\n`);
console.log(`Dashboard built in dist/. Backend: ${apiBaseUrl || 'same origin (/api)'}`);
