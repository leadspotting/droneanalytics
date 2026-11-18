export function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export function buildImagePreviewUrl(missionId, previewScale, previewMode, force = false) {
  const basePath = previewMode === 'grouped'
    ? `/image_mission/${encodeURIComponent(missionId)}/mosaic/preview/grouped`
    : `/image_mission/${encodeURIComponent(missionId)}/mosaic/preview`;

  const forceParam = force ? '&force=true' : '';
  const refinementParam = previewMode === 'grouped' ? '' : '&use_alignment_refinement=true';
  return `${basePath}?scale=${previewScale}&normalize_altitude=true${refinementParam}${forceParam}&t=${Date.now()}`;
}

export function buildImageExportUrl(missionId, previewScale) {
  return `/image_mission/${encodeURIComponent(missionId)}/mosaic/export?scale=${previewScale}&normalize_altitude=true`;
}

export function buildImagePreviewJob(missionId, previewScale, previewMode, force = false) {
  const grouped = previewMode === 'grouped';
  return {
    kind: grouped ? 'preview_grouped' : 'preview',
    mission_id: missionId,
    params: {
      scale: previewScale,
      normalize_altitude: true,
      ...(grouped ? {} : { use_alignment_refinement: true }),
      ...(force ? { force: true } : {}),
    },
  };
}

export function buildImageExportJob(missionId, previewScale) {
  return {
    kind: 'export',
    mission_id: missionId,
    params: { scale: previewScale, normalize_altitude: true },
  };
}
