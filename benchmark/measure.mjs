#!/usr/bin/env node

import { createServer } from 'node:http';
import { readFile, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const root = process.cwd();
const output = join(root, 'out');
const results = join(root, 'benchmark', 'results');
const port = Number(process.env.BENCHMARK_PORT ?? 4317);
const routes = ['/', '/content/about/', '/content/projects/', '/content/reports/', '/content/posts/'];
const budgets = { initialJs: 200_000, initialCss: 80_000, homepage: 2_000_000, route: 3_000_000, largest: 1_000_000 };

async function filesIn(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await filesIn(path));
    else files.push(path);
  }
  return files;
}

function contentType(path) {
  return { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.json': 'application/json', '.svg': 'image/svg+xml' }[extname(path)] ?? 'application/octet-stream';
}

function server() {
  return createServer(async (request, response) => {
    const requested = decodeURIComponent((request.url ?? '/').split('?')[0]);
    const candidates = requested.endsWith('/')
      ? [join(output, requested, 'index.html'), join(output, requested.slice(0, -1) + '.html')]
      : [join(output, requested), join(output, requested + '.html'), join(output, requested, 'index.html')];
    const path = candidates.find((candidate) => existsSync(candidate));
    if (!path) { response.writeHead(404); response.end('Not found'); return; }
    try {
      const body = await readFile(path);
      response.writeHead(200, { 'content-type': contentType(path), 'content-length': body.byteLength });
      response.end(body);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  });
}

async function request(path) {
  const started = performance.now();
  const response = await fetch(`http://127.0.0.1:${port}${path}`);
  const body = await response.arrayBuffer();
  return { path, status: response.status, bytes: body.byteLength, milliseconds: Math.round(performance.now() - started) };
}

async function main() {
  if (!existsSync(output)) throw new Error('out/ is missing; run npm run build first');
  const allFiles = await filesIn(output);
  const sizes = await Promise.all(allFiles.map(async (path) => ({ path: relative(output, path), bytes: (await stat(path)).size })));
  const js = sizes.filter((file) => file.path.endsWith('.js')).reduce((sum, file) => sum + file.bytes, 0);
  const css = sizes.filter((file) => file.path.endsWith('.css')).reduce((sum, file) => sum + file.bytes, 0);
  const largest = [...sizes].sort((a, b) => b.bytes - a.bytes).slice(0, 10);
  const http = server();
  await new Promise((resolve) => http.listen(port, '127.0.0.1', resolve));
  const responses = await Promise.all(routes.map(request));
  await new Promise((resolve) => http.close(resolve));
  const homepage = responses.find((item) => item.path === '/')?.bytes ?? 0;
  const checks = { initialJs: js <= budgets.initialJs, initialCss: css <= budgets.initialCss, homepage: homepage <= budgets.homepage, routes: responses.every((item) => item.bytes <= budgets.route && item.status === 200), largest: largest[0]?.bytes <= budgets.largest };
  const result = { generatedAt: new Date().toISOString(), node: process.version, budgets, totals: { files: sizes.length, exportBytes: sizes.reduce((sum, file) => sum + file.bytes, 0), javascriptBytes: js, cssBytes: css }, responses, largest, checks, passed: Object.values(checks).every(Boolean) };
  await import('node:fs/promises').then(({ mkdir, writeFile }) => mkdir(results, { recursive: true }).then(() => writeFile(join(results, 'latest.json'), JSON.stringify(result, null, 2))));
  console.log(JSON.stringify(result, null, 2));
  if (!result.passed && process.env.BENCHMARK_STRICT === '1') process.exitCode = 2;
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
