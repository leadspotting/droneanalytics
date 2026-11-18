async function readJson(response, fallbackMessage) {
  const rawBody = await response.text();
  let data = null;

  if (rawBody) {
    try {
      data = JSON.parse(rawBody);
    } catch {
      data = null;
    }
  }

  if (!response.ok) {
    throw new Error(data?.detail ?? rawBody ?? fallbackMessage);
  }

  if (data === null) {
    throw new Error(fallbackMessage);
  }

  return data;
}

export async function fetchMissions(options) {
  const response = await fetch('/missions', options);
  return readJson(response, 'Failed to load missions');
}

export async function fetchDrones(options) {
  const response = await fetch('/drones', options);
  return readJson(response, 'Failed to load drones');
}

export async function fetchMissionData(missionId, options) {
  const [footprintsResponse, heatmapResponse, mosaicResponse] = await Promise.all([
    fetch(`/mission/${encodeURIComponent(missionId)}/footprints`, options),
    fetch(`/mission/${encodeURIComponent(missionId)}/heatmap?resolution_m=300`, options),
    fetch(`/mission/${encodeURIComponent(missionId)}/mosaic`, options),
  ]);

  const [footprints, heatmap, mosaic] = await Promise.all([
    readJson(footprintsResponse, 'Failed to load footprints'),
    readJson(heatmapResponse, 'Failed to load heatmap'),
    readJson(mosaicResponse, 'Failed to load mosaic metadata'),
  ]);

  return { footprints, heatmap, mosaic };
}

export async function fetchImageMissions() {
  const response = await fetch('/image_missions');
  return readJson(response, 'Failed to load image missions');
}

export async function loadImageMission(payload) {
  const response = await fetch('/image_missions/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson(response, 'Failed to load image mission');
}

export async function uploadImageMissionFiles({ files, missionId, aglM, onProgress }) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  formData.append('mission_id', missionId);
  formData.append('agl_m', String(aglM));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/image_missions/upload');

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || typeof onProgress !== 'function') return;
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      onProgress(percent);
    };

    xhr.onerror = () => {
      reject(new Error('Failed to upload image mission'));
    };

    xhr.onload = () => {
      const response = {
        ok: xhr.status >= 200 && xhr.status < 300,
        status: xhr.status,
        text: async () => xhr.responseText,
      };

      readJson(response, 'Failed to upload image mission')
        .then(resolve)
        .catch(reject);
    };

    xhr.send(formData);
  });
}

export async function fetchImageMosaic(missionId) {
  const response = await fetch(`/image_mission/${encodeURIComponent(missionId)}/mosaic`);
  return readJson(response, 'Failed to load image mosaic');
}

export async function fetchImageMosaicDebug(missionId, force = false) {
  const params = new URLSearchParams();
  if (force) params.append('force', 'true');
  const response = await fetch(`/image_mission/${encodeURIComponent(missionId)}/mosaic/debug?${params.toString()}`);
  return readJson(response, 'Failed to fetch mosaic debug data');
}

export async function fetchTask4Reference() {
  const response = await fetch('/task4/reference');
  if (!response.ok) return null;
  return response.json();
}

export async function uploadTask4Reference(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/task4/reference/upload', {
    method: 'POST',
    body: formData,
  });

  return readJson(response, 'Failed to upload Task 4 reference');
}

export async function projectPixelToGps({ missionId, target, x, y, imageWidthPx }) {
  const basePath = target === 'image'
    ? `/image_mission/${encodeURIComponent(missionId)}/project`
    : `/mission/${encodeURIComponent(missionId)}/project`;
  const params = new URLSearchParams({
    x: String(x),
    y: String(y),
    image_width_px: String(imageWidthPx),
  });
  const response = await fetch(`${basePath}?${params.toString()}`);
  return readJson(response, 'Projection failed');
}

export async function projectGpsToPixel({ missionId, target, lat, lon, imageWidthPx }) {
  const basePath = target === 'image'
    ? `/image_mission/${encodeURIComponent(missionId)}/project/reverse`
    : `/mission/${encodeURIComponent(missionId)}/project/reverse`;
  const params = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    image_width_px: String(imageWidthPx),
  });
  const response = await fetch(`${basePath}?${params.toString()}`);
  return readJson(response, 'Reverse projection failed');
}

export async function uploadMissionCsv(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/upload', {
    method: 'POST',
    body: formData,
  });

  return readJson(response, 'Upload failed');
}

export async function createCustomDrone(formData) {
  const response = await fetch('/drones', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData),
  });
  return readJson(response, 'Failed to add drone');
}

export async function planMission(area) {
  const response = await fetch('/plan_mission', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      area,
      altitude_m: 100.0,
      overlap_h: 0.7,
      overlap_v: 0.7,
    }),
  });
  return readJson(response, 'Failed to plan mission');
}

export async function recommendFlightParams(payload) {
  const response = await fetch('/flight_params/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson(response, 'Failed to recommend flight parameters');
}

const JOB_POLL_MS = 500;

function abortableSleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      window.clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    }, { once: true });
  });
}

/**
 * Run a server-side render job (preview / export) and report its progress.
 * Resolves with the result URL once the job is done. Aborting the signal
 * cancels the job on the server.
 */
export async function runRenderJob(spec, { onProgress, signal } = {}) {
  const startResponse = await fetch('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
    signal,
  });
  let status = await readJson(startResponse, 'Failed to start render');
  const jobId = status.job_id;

  signal?.addEventListener('abort', () => {
    fetch(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' }).catch(() => {});
  }, { once: true });

  while (status.status === 'queued' || status.status === 'running') {
    onProgress?.(status);
    await abortableSleep(JOB_POLL_MS, signal);
    const response = await fetch(`/jobs/${encodeURIComponent(jobId)}`, { signal });
    status = await readJson(response, 'Failed to read render progress');
  }
  onProgress?.(status);

  if (status.status !== 'done') {
    throw new Error(status.error || `Render ${status.status}`);
  }
  return status.result_url;
}
