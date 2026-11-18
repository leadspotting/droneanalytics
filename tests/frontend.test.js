import test from 'node:test';
import assert from 'node:assert/strict';
import { escapeHtml } from '../dashboard/src/utils/html.js';
import { buildImageExportUrl, buildImagePreviewUrl } from '../dashboard/src/utils/imageMission.js';

test('CSV popup text cannot inject markup or event handlers', () => {
  assert.equal(escapeHtml('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
  assert.equal(escapeHtml("Tom & 'Jerry'"), 'Tom &amp; &#39;Jerry&#39;');
  assert.equal(escapeHtml(null), '');
});

test('image routes preserve mission IDs containing URL delimiters', () => {
  assert.equal(buildImageExportUrl('Mission #1?', 0.12), '/image_mission/Mission%20%231%3F/mosaic/export?scale=0.12&normalize_altitude=true');
  assert.match(buildImagePreviewUrl('Mission #1?', 0.12, 'single'), /^\/image_mission\/Mission%20%231%3F\/mosaic\/preview\?/);
});
