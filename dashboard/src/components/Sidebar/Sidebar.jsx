import { useEffect, useState } from 'react';
import GsdCalculator from './GsdCalculator';
import Stats from './Stats';
import { TaskSectionHeader } from './TaskGuide';
import { runRenderJob } from '../../services/api';

export default function Sidebar({
  fpData,
  hmData,
  layers,
  onLayerToggle,
  lang,
  t,
  activeWaypoint,
  onWaypointClick,
  draftMission,
  savedMission,
  hasUnsavedChanges,
  canUndo,
  onSaveDraft,
  onUndo,
  drones,
  selectedDrone,
  onDroneSelect,
  onAddCustomDrone,
  feasibility,
  isFeasible,
  selectedMission,
  mosaicMeta,
  imageMissions,
  selectedImageMission,
  onSelectImageMission,
  imageMosaicMeta,
  imagePreviewUrl,
  imagePreviewScale,
  imagePreviewMode,
  onImagePreviewModeChange,
  imagePreviewJob,
  onCancelImagePreview,
  imageExportJob,
  onLoadDefaultImageMission,
  onUploadImageMission,
  task4ReferenceMeta,
  task4ReferencePreviewUrl,
  onImagePreviewPick,
  onTask4PreviewPick,
  onTask4ReferenceUpload,
  onRecommendFlightParams,
  onProjectPixelToGps,
  onProjectGpsToPixel,
  onReRender,
}) {
  const fps = fpData?.footprints ?? [];

  const [showCustomForm, setShowCustomForm] = useState(false);
  const [customForm, setCustomForm] = useState({
    model: '',
    series: 'Custom',
    cruise_speed_kmh: '',
    max_speed_kmh: '',
    max_flight_time_min: '',
  });
  const [customError, setCustomError] = useState('');

  const [projectionTarget, setProjectionTarget] = useState('mission');
  const [projectionForm, setProjectionForm] = useState({
    imageWidthPx: 4000,
    x: 0,
    y: 0,
    lat: '',
    lon: '',
  });
  const [projectionResult, setProjectionResult] = useState(null);
  const [projectionError, setProjectionError] = useState('');
  const [selectedImagePreviewPoint, setSelectedImagePreviewPoint] = useState(null);
  const [selectedTask4PreviewPoint, setSelectedTask4PreviewPoint] = useState(null);
  const [imagePreviewLoading, setImagePreviewLoading] = useState(false);
  const [isImagePreviewModalOpen, setIsImagePreviewModalOpen] = useState(false);
  const [imageUploadLoading, setImageUploadLoading] = useState(false);
  const [imageUploadProgress, setImageUploadProgress] = useState(0);
  const [imageUploadError, setImageUploadError] = useState('');
  const [imageMissionForm, setImageMissionForm] = useState({
    missionId: '',
    aglM: '110',
  });
  const [task4UploadLoading, setTask4UploadLoading] = useState(false);
  const [task4UploadError, setTask4UploadError] = useState('');
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [downloadError, setDownloadError] = useState('');
  const [exportJob, setExportJob] = useState(null);

  useEffect(() => {
    setImagePreviewLoading(Boolean(imagePreviewUrl));
  }, [imagePreviewUrl]);

  useEffect(() => {
    if (!imagePreviewUrl) {
      setIsImagePreviewModalOpen(false);
    }
  }, [imagePreviewUrl]);

  useEffect(() => {
    setImageUploadError('');
    setImageUploadProgress(0);
  }, [selectedImageMission]);

  useEffect(() => {
    setSelectedTask4PreviewPoint(null);
    setTask4UploadError('');
  }, [task4ReferencePreviewUrl]);

  const customFormValid =
    customForm.model.trim() &&
    Number(customForm.cruise_speed_kmh) > 0 &&
    Number(customForm.max_speed_kmh) > 0 &&
    Number(customForm.max_flight_time_min) > 0;

  const imageWidthPx = Number(projectionForm.imageWidthPx) || 4000;
  const activeProjectionMeta = projectionTarget === 'image' ? imageMosaicMeta : mosaicMeta;
  const activeProjectionMission = projectionTarget === 'image' ? selectedImageMission : selectedMission;
  const normalisedImagePreviewPoint = selectedImagePreviewPoint && imageMosaicMeta
    ? {
        x: selectedImagePreviewPoint.x / Math.max(imageMosaicMeta.canvas_width_px - 1, 1),
        y: selectedImagePreviewPoint.y / Math.max(imageMosaicMeta.canvas_height_px - 1, 1),
      }
    : null;
  const normalisedTask4PreviewPoint = selectedTask4PreviewPoint && task4ReferenceMeta
    ? {
        x: selectedTask4PreviewPoint.x / Math.max(task4ReferenceMeta.width_px - 1, 1),
        y: selectedTask4PreviewPoint.y / Math.max(task4ReferenceMeta.height_px - 1, 1),
      }
    : null;

  function setImageSelection(x, y) {
    setSelectedImagePreviewPoint({ x, y });
  }

  function setTask4Selection(x, y) {
    setSelectedTask4PreviewPoint({ x, y });
  }

  function getProjectionActionLabel(result) {
    if (!result) return '';
    if (result.mode === 'gps_to_pixel') {
      return lang === 'he' ? 'הוקרן חזרה לפיקסל' : 'Projected back to pixel';
    }
    if (result.source === 'image_click') {
      return lang === 'he' ? 'נבחר מהתמונה' : 'Selected from image';
    }
    return lang === 'he' ? 'הוקרן ל-GPS' : 'Projected to GPS';
  }

  async function handleCustomSubmit() {
    setCustomError('');
    try {
      await onAddCustomDrone({
        model: customForm.model.trim(),
        series: customForm.series.trim() || 'Custom',
        cruise_speed_kmh: Number(customForm.cruise_speed_kmh),
        max_speed_kmh: Number(customForm.max_speed_kmh),
        max_flight_time_min: Number(customForm.max_flight_time_min),
      });
      setShowCustomForm(false);
      setCustomForm({
        model: '',
        series: 'Custom',
        cruise_speed_kmh: '',
        max_speed_kmh: '',
        max_flight_time_min: '',
      });
    } catch (e) {
      setCustomError(e.message ?? 'Failed to add drone');
    }
  }

  async function handlePixelToGps() {
    setProjectionError('');
    try {
      const data = await onProjectPixelToGps({
        x: Number(projectionForm.x),
        y: Number(projectionForm.y),
        imageWidthPx,
        target: projectionTarget,
        missionId: activeProjectionMission,
      });
      if (projectionTarget === 'image') {
        setImageSelection(data.x, data.y);
      }
      setProjectionResult({ mode: 'pixel_to_gps', source: 'manual', data });
      setProjectionForm((form) => ({ ...form, lat: data.lat, lon: data.lon }));
    } catch (e) {
      setProjectionError(e.message ?? 'Projection failed');
    }
  }

  async function handleGpsToPixel() {
    setProjectionError('');
    try {
      const data = await onProjectGpsToPixel({
        lat: Number(projectionForm.lat),
        lon: Number(projectionForm.lon),
        imageWidthPx,
        target: projectionTarget,
        missionId: activeProjectionMission,
      });
      if (projectionTarget === 'image') {
        setImageSelection(data.x, data.y);
      }
      setProjectionResult({ mode: 'gps_to_pixel', data });
      setProjectionForm((form) => ({ ...form, x: data.x, y: data.y }));
    } catch (e) {
      setProjectionError(e.message ?? 'Reverse projection failed');
    }
  }

  function handleProjectionField(key, value) {
    setProjectionForm((form) => ({ ...form, [key]: value }));
  }

  function fillProjectionPreset(kind) {
    if (!activeProjectionMeta) return;
    const presets = {
      tl: { x: 0, y: 0 },
      center: {
        x: Math.floor(activeProjectionMeta.canvas_width_px / 2),
        y: Math.floor(activeProjectionMeta.canvas_height_px / 2),
      },
      br: {
        x: Math.max(activeProjectionMeta.canvas_width_px - 1, 0),
        y: Math.max(activeProjectionMeta.canvas_height_px - 1, 0),
      },
    };
    if (projectionTarget === 'image') {
      setImageSelection(presets[kind].x, presets[kind].y);
    }
    setProjectionForm((form) => ({ ...form, ...presets[kind] }));
  }

  async function handleImagePreviewClick(e) {
    if (!imageMosaicMeta) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const xNorm = (e.clientX - rect.left) / rect.width;
    const yNorm = (e.clientY - rect.top) / rect.height;
    const x = Math.max(0, Math.min(imageMosaicMeta.canvas_width_px - 1, Math.round(xNorm * (imageMosaicMeta.canvas_width_px - 1))));
    const y = Math.max(0, Math.min(imageMosaicMeta.canvas_height_px - 1, Math.round(yNorm * (imageMosaicMeta.canvas_height_px - 1))));
    setProjectionTarget('image');
    setProjectionForm((form) => ({
      ...form,
      imageWidthPx: imageMosaicMeta.image_width_px ?? 4928,
      x,
      y,
    }));
    setImageSelection(x, y);
    setProjectionError('');
    try {
      const data = await onImagePreviewPick?.({
        x,
        y,
        imageWidthPx: imageMosaicMeta.image_width_px ?? 4928,
        target: 'image',
        missionId: selectedImageMission,
      });
      if (data) {
        setProjectionResult({ mode: 'pixel_to_gps', source: 'image_click', data });
        setProjectionForm((form) => ({ ...form, lat: data.lat, lon: data.lon }));
      }
    } catch (e2) {
      setProjectionError(e2.message ?? 'Projection failed');
    }
  }

  async function handleTask4PreviewClick(e) {
    if (!task4ReferenceMeta) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const xNorm = (e.clientX - rect.left) / rect.width;
    const yNorm = (e.clientY - rect.top) / rect.height;
    const x = Math.max(0, Math.min(task4ReferenceMeta.width_px - 1, Math.round(xNorm * (task4ReferenceMeta.width_px - 1))));
    const y = Math.max(0, Math.min(task4ReferenceMeta.height_px - 1, Math.round(yNorm * (task4ReferenceMeta.height_px - 1))));
    setTask4Selection(x, y);
    setProjectionError('');

    try {
      const data = await onTask4PreviewPick?.({ x, y, xNorm, yNorm });
      if (data) {
        setProjectionResult({ mode: 'pixel_to_gps', source: 'task4_click', data });
      }
    } catch (error) {
      setProjectionError(error.message ?? 'Task 4 projection failed');
    }
  }

  async function handleTask4FileChange(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    setTask4UploadLoading(true);
    setTask4UploadError('');
    try {
      await onTask4ReferenceUpload?.(file);
    } catch (error) {
      setTask4UploadError(error.message ?? 'Failed to upload GeoTIFF');
    } finally {
      setTask4UploadLoading(false);
    }
  }

  async function handleImageMissionFilesChange(event) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = '';
    if (files.length === 0) return;

    setImageUploadLoading(true);
    setImageUploadProgress(0);
    setImageUploadError('');
    try {
      await onUploadImageMission?.({
        files,
        missionId: imageMissionForm.missionId,
        aglM: Number(imageMissionForm.aglM) || 110,
        onProgress: setImageUploadProgress,
      });
      setImageUploadProgress(100);
    } catch (error) {
      setImageUploadError(error.message ?? 'Failed to upload image mission');
      setImageUploadProgress(0);
    } finally {
      setImageUploadLoading(false);
    }
  }

  async function handleDownloadImageExport() {
    if (!imageExportJob || downloadLoading) return;
    setDownloadError('');
    setDownloadLoading(true);
    try {
      const resultUrl = await runRenderJob(imageExportJob, { onProgress: setExportJob });
      const response = await fetch(resultUrl);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      // Derive filename from the Content-Disposition header or fall back to mission id
      const disposition = response.headers.get('Content-Disposition') ?? '';
      const match = disposition.match(/filename="?([^";\n]+)"?/i);
      link.download = match ? match[1] : `mosaic_export.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setDownloadError(lang === 'he' ? `שגיאה: ${err.message}` : `Download failed: ${err.message}`);
    } finally {
      setDownloadLoading(false);
      setExportJob(null);
    }
  }

  const dronesBySeries = (drones ?? []).reduce((acc, d) => {
    (acc[d.series] = acc[d.series] ?? []).push(d);
    return acc;
  }, {});
  const builtinSeries = Object.keys(dronesBySeries).filter((s) => s !== 'Custom');
  const customDrones = dronesBySeries.Custom ?? [];

  const alerts = [];
  fps.forEach((fp, i) => {
    if (fp.gps_tampering === '1') alerts.push({ cls: 'danger', icon: 'GPS', text: `WP ${i + 1}: GPS Tampered` });
    if (fp.data_leak === '1') alerts.push({ cls: 'danger', icon: 'Leak', text: `WP ${i + 1}: Data Leak` });
    if (fp.memory_integrity !== 'OK') alerts.push({ cls: 'warn', icon: 'Mem', text: `WP ${i + 1}: ${fp.memory_integrity}` });
  });

  const layerDefs = [
    { key: 'path', label: t('layerPath'), id: 'togglePath' },
    { key: 'footprints', label: t('layerFootprints'), id: 'toggleFP' },
    { key: 'heatmap', label: t('layerHeatmap'), id: 'toggleHM' },
    { key: 'markers', label: t('layerWP'), id: 'toggleWP' },
  ];

  const saveDisabledReason = !selectedDrone
    ? (t('planSelectDrone') || 'Select a drone first')
    : !isFeasible
      ? (t('planInfeasible') || 'Mission exceeds safe battery')
      : null;

  const exportDisabledReason = hasUnsavedChanges
    ? (lang === 'he' ? 'שמור קודם את עריכות נקודות הציון' : 'Save waypoint edits before exporting')
    : !selectedDrone
      ? (t('planSelectDrone') || 'Select a drone first')
      : !isFeasible
        ? (t('planInfeasible') || 'Mission exceeds safe battery')
        : null;

  return (
    <div className="panel">
      <div className="panel-section">
        <TaskSectionHeader lang={lang} taskKey="footprint" />
        <TaskSectionHeader lang={lang} taskKey="heatmap" />
        <h3>{t('pnlOverview')}</h3>
        <Stats fpData={fpData} hmData={hmData} t={t} />
      </div>

      <div className="panel-section">
        <TaskSectionHeader lang={lang} taskKey="altitude" />
        <h3>{lang === 'he' ? 'פרמטרי טיסה' : 'Flight Parameters'}</h3>
        <GsdCalculator
          lang={lang}
          selectedDrone={selectedDrone}
          onRecommendFlightParams={onRecommendFlightParams}
        />
      </div>


      {draftMission && (
        <div
          className="panel-section"
          style={{
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '8px',
            padding: '12px',
        }}
        >
          <TaskSectionHeader lang={lang} taskKey="planner" />
          <h3 style={{ color: '#fbbf24', marginTop: 0 }}>{t('planTitle') || 'Mission Planner'}</h3>

          {hasUnsavedChanges && (
            <p style={{ margin: '0 0 8px', fontSize: '11px', color: '#f59e0b', fontWeight: 600 }}>
              {lang === 'he' ? 'שינויים שלא נשמרו' : 'Unsaved changes'}
            </p>
          )}

          <div style={{ marginBottom: '8px' }}>
            <label style={{ display: 'block', fontSize: '11px', color: 'var(--muted)', marginBottom: '3px' }}>
              {t('planSelectDrone') || 'Drone Model'}
            </label>
            <select
              value={selectedDrone?.id ?? ''}
              onChange={(e) => {
                if (e.target.value === '__add_custom__') setShowCustomForm(true);
                else onDroneSelect(e.target.value || null);
              }}
              style={{
                width: '100%',
                padding: '6px 8px',
                fontSize: '13px',
                backgroundColor: 'var(--surface, #1e1e2e)',
                border: '1px solid rgba(245,158,11,0.4)',
                borderRadius: '4px',
                color: 'inherit',
                cursor: 'pointer',
              }}
            >
              <option value="" disabled>{lang === 'he' ? 'בחר דגם' : 'Select drone'}</option>
              {builtinSeries.map((series) => (
                <optgroup key={series} label={series}>
                  {dronesBySeries[series].map((d) => (
                    <option key={d.id} value={d.id}>{d.model}</option>
                  ))}
                </optgroup>
              ))}
              {customDrones.length > 0 && (
                <optgroup label={lang === 'he' ? 'מותאם אישית' : 'Custom'}>
                  {customDrones.map((d) => (
                    <option key={d.id} value={d.id}>{d.model}</option>
                  ))}
                </optgroup>
              )}
              <option value="__add_custom__">{t('planAddCustom') || 'Add custom drone'}</option>
            </select>
          </div>

          {showCustomForm && (
            <div
              style={{
                margin: '8px 0',
                padding: '10px',
                border: '1px solid rgba(139,92,246,0.4)',
                borderRadius: '6px',
                backgroundColor: 'rgba(139,92,246,0.08)',
                fontSize: '12px',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: '6px', color: '#a78bfa' }}>
                {t('planCustomTitle') || 'Add Custom Drone'}
              </div>
              {[
                { key: 'model', label: t('planModelName') || 'Model name', ph: 'My Drone X1', type: 'text' },
                { key: 'series', label: lang === 'he' ? 'סדרה' : 'Series', ph: 'Custom', type: 'text' },
                { key: 'cruise_speed_kmh', label: t('planCruiseSpd') || 'Cruise speed (km/h)', ph: '54', type: 'number' },
                { key: 'max_speed_kmh', label: t('planMaxSpd') || 'Max speed (km/h)', ph: '75', type: 'number' },
                { key: 'max_flight_time_min', label: t('planMaxTime') || 'Max flight time (min)', ph: '30', type: 'number' },
              ].map(({ key, label, ph, type }) => (
                <div key={key} style={{ marginBottom: '5px' }}>
                  <label style={{ display: 'block', color: 'var(--muted)', marginBottom: '2px' }}>{label}</label>
                  <input
                    type={type}
                    value={customForm[key]}
                    placeholder={ph}
                    onChange={(e) => setCustomForm((f) => ({ ...f, [key]: e.target.value }))}
                    style={{
                      width: '100%',
                      padding: '4px 6px',
                      fontSize: '12px',
                      backgroundColor: 'var(--surface, #1e1e2e)',
                      border: '1px solid rgba(139,92,246,0.3)',
                      borderRadius: '3px',
                      color: 'inherit',
                    }}
                  />
                </div>
              ))}
              {customError && <p style={{ color: '#f87171', fontSize: '11px', margin: '4px 0' }}>{customError}</p>}
              <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                <button
                  type="button"
                  onClick={handleCustomSubmit}
                  disabled={!customFormValid}
                  style={{
                    flex: 1,
                    padding: '5px',
                    fontSize: '12px',
                    cursor: customFormValid ? 'pointer' : 'not-allowed',
                    backgroundColor: customFormValid ? '#7c3aed' : '#4b5563',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                  }}
                >
                  {lang === 'he' ? 'הוסף' : 'Add'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowCustomForm(false);
                    setCustomError('');
                  }}
                  style={{
                    flex: 1,
                    padding: '5px',
                    fontSize: '12px',
                    cursor: 'pointer',
                    backgroundColor: 'transparent',
                    border: '1px solid rgba(139,92,246,0.4)',
                    borderRadius: '4px',
                    color: '#a78bfa',
                  }}
                >
                  {lang === 'he' ? 'בטל' : 'Cancel'}
                </button>
              </div>
            </div>
          )}

          {selectedDrone && feasibility && (
            <div
              style={{
                margin: '8px 0',
                padding: '8px 10px',
                border: `1px solid ${feasibility.feasible ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`,
                borderRadius: '6px',
                backgroundColor: feasibility.feasible ? 'rgba(34,197,94,0.07)' : 'rgba(239,68,68,0.07)',
                fontSize: '12px',
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: '5px', color: feasibility.feasible ? '#4ade80' : '#f87171' }}>
                {feasibility.feasible ? (t('planFeasible') || 'Mission feasible') : (t('planInfeasible') || 'Exceeds safe battery')}
              </div>
              <div style={{ color: 'var(--muted)', marginBottom: '4px', fontSize: '11px' }}>
                {selectedDrone.model} · {t('planDroneSpd') || 'Cruise'}: {selectedDrone.cruise_speed_kmh} km/h
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 10px' }}>
                <div><span style={{ color: 'var(--muted)' }}>{t('planSafeTime') || 'Safe limit'}: </span><strong>{feasibility.safe_flight_time_min} min</strong></div>
                <div><span style={{ color: 'var(--muted)' }}>{t('planEstTime') || 'Est. time'}: </span><strong>{feasibility.estimated_time_min} min</strong></div>
                <div><span style={{ color: 'var(--muted)' }}>{t('planDistance') || 'Route'}: </span><strong>{feasibility.distance_km} km</strong></div>
                <div><span style={{ color: 'var(--muted)' }}>{t('planBattery') || 'Battery'}: </span><strong>{feasibility.battery_usage_pct}%</strong></div>
              </div>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '14px', marginBottom: '8px' }}>
            <div>
              <span style={{ color: 'var(--muted)' }}>{t('planWpCount') || 'Waypoints'}:</span>
              <br /><strong>{draftMission.waypoints?.length ?? draftMission.total_waypoints}</strong>
            </div>
            <div>
              <span style={{ color: 'var(--muted)' }}>{t('planFlightTime') || 'Est. Time'}:</span>
              <br /><strong>{draftMission.flight_time_sec} sec</strong>
            </div>
          </div>

          <p style={{ margin: '0 0 8px', fontSize: '12px', color: 'var(--muted)' }}>
            {t('planSubtitle') || 'Draw a polygon on the map to re-plan.'}
          </p>

          {(canUndo || hasUnsavedChanges) && (
            <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
              {canUndo && (
                <button
                  type="button"
                  onClick={onUndo}
                  style={{
                    flex: 1,
                    padding: '6px',
                    cursor: 'pointer',
                    backgroundColor: 'transparent',
                    border: '1px solid rgba(245,158,11,0.5)',
                    borderRadius: '4px',
                    color: '#f59e0b',
                    fontSize: '13px',
                  }}
                >
                  {lang === 'he' ? 'בטל' : 'Undo'}
                </button>
              )}
              {hasUnsavedChanges && (
                <button
                  type="button"
                  onClick={isFeasible ? onSaveDraft : undefined}
                  disabled={!isFeasible}
                  title={saveDisabledReason ?? 'Save'}
                  style={{
                    flex: 1,
                    padding: '6px',
                    cursor: isFeasible ? 'pointer' : 'not-allowed',
                    backgroundColor: isFeasible ? '#22c55e' : '#4b5563',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    fontWeight: 'bold',
                    fontSize: '13px',
                    opacity: isFeasible ? 1 : 0.6,
                  }}
                >
                  {lang === 'he' ? 'שמור' : 'Save'}
                </button>
              )}
            </div>
          )}

          <button
            type="button"
            onClick={() => {
              if (!savedMission?.waypoints || exportDisabledReason) return;
              let csv = 'data:text/csv;charset=utf-8,Waypoint_Number,Latitude,Longitude,Altitude_m\n';
              savedMission.waypoints.forEach((wp, idx) => {
                csv += `${idx + 1},${wp.lat.toFixed(6)},${wp.lon.toFixed(6)},${wp.alt.toFixed(1)}\n`;
              });
              const link = document.createElement('a');
              link.setAttribute('href', encodeURI(csv));
              link.setAttribute('download', 'flight_plan.csv');
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
            }}
            disabled={!!exportDisabledReason}
            title={exportDisabledReason ?? 'Export flight plan'}
            style={{
              width: '100%',
              padding: '8px',
              cursor: exportDisabledReason ? 'not-allowed' : 'pointer',
              backgroundColor: exportDisabledReason ? '#4b5563' : '#f59e0b',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              fontWeight: 'bold',
              opacity: exportDisabledReason ? 0.5 : 1,
            }}
          >
            {t('planExport') || 'Export to Drone (CSV)'}
          </button>
        </div>
      )}

      <div className="panel-section">
        <TaskSectionHeader lang={lang} taskKey="mosaic" />
        <h3>{lang === 'he' ? 'תצוגת מוזאיקה' : 'Mosaic Preview'}</h3>
        <div className="projection-stack">
          <button type="button" className="projection-btn" onClick={onLoadDefaultImageMission}>
            {lang === 'he' ? 'רענן רשימת משימות תמונה' : 'Refresh image missions'}
          </button>
          <div className="projection-grid">
            <div>
              <label className="projection-label">
                {lang === 'he' ? 'מזהה משימה' : 'Mission ID'}
              </label>
              <input
                type="text"
                value={imageMissionForm.missionId}
                onChange={(event) => setImageMissionForm((current) => ({ ...current, missionId: event.target.value }))}
                className="projection-input"
                placeholder={lang === 'he' ? 'לדוגמה: FIELD_UPLOAD' : 'Example: FIELD_UPLOAD'}
              />
            </div>
            <div>
              <label className="projection-label">
                {lang === 'he' ? 'גובה AGL (מ׳)' : 'AGL (m)'}
              </label>
              <input
                type="number"
                min="1"
                value={imageMissionForm.aglM}
                onChange={(event) => setImageMissionForm((current) => ({ ...current, aglM: event.target.value }))}
                className="projection-input"
              />
            </div>
          </div>
          <label className="projection-btn projection-btn-secondary task4-upload-btn">
            <input
              type="file"
              accept=".jpg,.jpeg"
              multiple
              className="task4-upload-input"
              onChange={handleImageMissionFilesChange}
            />
            {imageUploadLoading
              ? (lang === 'he' ? 'מעלה תמונות...' : 'Uploading images...')
              : (lang === 'he' ? 'העלה תמונות מהמחשב' : 'Upload images from computer')}
          </label>
          <div className="projection-note">
            {lang === 'he'
              ? 'נתמך: קבצי JPG/JPEG עם GPS EXIF.'
              : 'Supported: JPG/JPEG files with GPS EXIF.'}
          </div>
          {imageUploadLoading && (
            <div className="upload-progress">
              <div className="upload-progress-row">
                <span>{lang === 'he' ? 'מעלה קבצים' : 'Uploading files'}</span>
                <strong>{imageUploadProgress}%</strong>
              </div>
              <div className="upload-progress-track">
                <div
                  className="upload-progress-bar"
                  style={{ width: `${imageUploadProgress}%` }}
                />
              </div>
            </div>
          )}
          {imageUploadError && <div className="projection-error">{imageUploadError}</div>}
          <select value={selectedImageMission} onChange={(e) => onSelectImageMission(e.target.value)}>
            <option value="">{lang === 'he' ? 'בחר image mission' : 'Select image mission'}</option>
            {imageMissions.map((m) => (
              <option key={m.mission_id} value={m.mission_id}>
                {m.mission_id} ({m.image_count})
              </option>
            ))}
          </select>
          <div className="projection-chips">
            <button
              type="button"
              className={`projection-chip${imagePreviewMode === 'single' ? ' projection-chip-active' : ''}`}
              onClick={() => onImagePreviewModeChange?.('single')}
            >
              {lang === 'he' ? 'בודד' : 'Single'}
            </button>
            <button
              type="button"
              className={`projection-chip${imagePreviewMode === 'grouped' ? ' projection-chip-active' : ''}`}
              onClick={() => onImagePreviewModeChange?.('grouped')}
            >
              {lang === 'he' ? 'מקובץ' : 'Grouped'}
            </button>
          </div>
          <div className="projection-note">
            {lang === 'he'
              ? 'מצב התצוגה עובד לכל משימת תמונות. בחר תצוגה שמתאימה לבדיקה שלך.'
              : 'Preview mode works for any image mission. Choose the mode that fits your review workflow.'}
          </div>
          {imagePreviewJob && (
            <div className="upload-progress">
              <div className="upload-progress-row">
                <span>
                  {imagePreviewJob.status === 'error'
                    ? (lang === 'he' ? 'הרינדור נכשל' : 'Render failed')
                    : imagePreviewJob.status === 'queued' && imagePreviewJob.queue_position > 0
                      ? (lang === 'he' ? `בתור (${imagePreviewJob.queue_position})` : `Queued (${imagePreviewJob.queue_position} ahead)`)
                      : (imagePreviewJob.stage || (lang === 'he' ? 'מעבד...' : 'Rendering...'))}
                </span>
                {imagePreviewJob.status !== 'error' && (
                  <strong>{Math.round((imagePreviewJob.progress ?? 0) * 100)}%</strong>
                )}
              </div>
              {imagePreviewJob.status === 'error' ? (
                <div className="projection-error">{imagePreviewJob.error}</div>
              ) : (
                <>
                  <div className="upload-progress-track">
                    <div
                      className="upload-progress-bar"
                      style={{ width: `${Math.round((imagePreviewJob.progress ?? 0) * 100)}%` }}
                    />
                  </div>
                  <button
                    type="button"
                    className="projection-btn projection-btn-secondary"
                    onClick={onCancelImagePreview}
                  >
                    {lang === 'he' ? 'ביטול' : 'Cancel'}
                  </button>
                </>
              )}
            </div>
          )}
          {imagePreviewUrl ? (
            <>
              <div className="projection-preview-frame">
                <img
                  src={imagePreviewUrl}
                  alt="Mosaic preview"
                  className="projection-preview"
                  onLoad={() => setImagePreviewLoading(false)}
                  onError={() => setImagePreviewLoading(false)}
                  onClick={handleImagePreviewClick}
                />
                {imagePreviewLoading && (
                  <div className="projection-preview-loading">
                    <div className="projection-preview-spinner" />
                    <span>{lang === 'he' ? 'טוען preview...' : 'Loading preview...'}</span>
                  </div>
                )}
                {normalisedImagePreviewPoint && (
                  <div
                    className="projection-preview-marker"
                    style={{
                      left: `${normalisedImagePreviewPoint.x * 100}%`,
                      top: `${normalisedImagePreviewPoint.y * 100}%`,
                    }}
                  />
                )}
              </div>
              <div className="projection-note">
                {lang === 'he' ? 'לחץ על התמונה כדי למלא X,Y' : 'Click the image to fill X,Y'}
                {imagePreviewScale ? ` · scale ${imagePreviewScale}` : ''}
              </div>
              {imageMosaicMeta && (
                <div className="projection-note">
                  <div>{lang === 'he' ? 'קנבס' : 'Canvas'}: {imageMosaicMeta.canvas_width_px} x {imageMosaicMeta.canvas_height_px}</div>
                  <div>GSD: {imageMosaicMeta.gsd_m} m/px</div>
                </div>
              )}
              <button
                type="button"
                className="projection-btn projection-btn-secondary"
                onClick={onReRender}
                disabled={!selectedImageMission || imagePreviewJob?.status === 'queued' || imagePreviewJob?.status === 'running'}
                title={lang === 'he' ? 'חישוב מחדש של המוזאיקה' : 'Re-stitch the mosaic'}
              >
                {lang === 'he' ? '🔄 רינדור מחדש' : '🔄 Re-render'}
              </button>
              <button
                type="button"
                className="projection-btn projection-btn-secondary"
                onClick={handleDownloadImageExport}
                disabled={!imageExportJob || downloadLoading}
              >
                {downloadLoading
                  ? `${lang === 'he' ? 'מכין' : 'Preparing'} ${Math.round((exportJob?.progress ?? 0) * 100)}%`
                  : (lang === 'he' ? 'הורד Export' : 'Download Export')}
              </button>
              {downloadError && (
                <div className="projection-error">{downloadError}</div>
              )}
              <button
                type="button"
                className="projection-btn projection-btn-secondary"
                onClick={() => setIsImagePreviewModalOpen(true)}
              >
                {lang === 'he' ? 'פתח תצוגה גדולה' : 'Open large preview'}
              </button>
            </>
          ) : (
            <div className="projection-note">
              {lang === 'he' ? 'טען image mission כדי לראות את התמונה' : 'Load an image mission to see the image'}
            </div>
          )}
        </div>
      </div>

      <div className="panel-section">
        <TaskSectionHeader lang={lang} taskKey="reference" />
        <div className="projection-stack">
          <label className="projection-btn projection-btn-secondary task4-upload-btn">
            <input
              type="file"
              accept=".tif,.tiff,.jpg,.jpeg"
              className="task4-upload-input"
              onChange={handleTask4FileChange}
            />
            {task4UploadLoading
              ? (lang === 'he' ? 'מעלה קובץ גיאוגרפי...' : 'Uploading geospatial file...')
              : (lang === 'he' ? 'העלה תמונה גיאוגרפית' : 'Upload geospatial image')}
          </label>
          <div className="projection-note">
            {lang === 'he'
              ? 'נתמך: GeoTIFF או JPG עם GPS EXIF.'
              : 'Supported: GeoTIFF or JPG with GPS EXIF.'}
          </div>
          {task4UploadError && <div className="projection-error">{task4UploadError}</div>}
          {task4ReferencePreviewUrl ? (
            <>
              <div className="projection-preview-frame projection-preview-reference">
                <img
                  src={task4ReferencePreviewUrl}
                  alt="Task 4 reference orthomosaic"
                  className="projection-preview"
                  onClick={handleTask4PreviewClick}
                />
                {normalisedTask4PreviewPoint && (
                  <div
                    className="projection-preview-marker projection-preview-marker-reference"
                    style={{
                      left: `${normalisedTask4PreviewPoint.x * 100}%`,
                      top: `${normalisedTask4PreviewPoint.y * 100}%`,
                    }}
                  />
                )}
              </div>
              {task4ReferenceMeta && (
                <div className="projection-note">
                  <div>{lang === 'he' ? 'קובץ' : 'File'}: {task4ReferenceMeta.filename}</div>
                  <div>{lang === 'he' ? 'גודל' : 'Size'}: {task4ReferenceMeta.width_px} x {task4ReferenceMeta.height_px}</div>
                  <div>GSD: {task4ReferenceMeta.gsd_m} m/px</div>
                  <div>{lang === 'he' ? 'סוג' : 'Type'}: {task4ReferenceMeta.reference_type === 'jpg_exif'
                    ? 'JPG + EXIF'
                    : 'GeoTIFF'}</div>
                  <div>{lang === 'he' ? 'מקור' : 'Source'}: {task4ReferenceMeta.source === 'uploaded'
                    ? (lang === 'he' ? 'הועלה ידנית' : 'Uploaded')
                    : (lang === 'he' ? 'מובנה בפרויקט' : 'Bundled')}</div>
                  {lang === 'he'
                    ? (task4ReferenceMeta.reference_type === 'jpg_exif'
                      ? 'JPG עם EXIF ימוקם לפי נתוני הרחפן והקליק יעבוד בקירוב על בסיס footprint'
                      : 'לחץ על תמונת הרפרנס כדי למקד את המפה לפי המיקום הגיאוגרפי שלה')
                    : (task4ReferenceMeta.reference_type === 'jpg_exif'
                      ? 'JPG with EXIF is mapped from drone metadata and clicks use an approximate footprint'
                      : 'Click the reference image to focus the map using its geospatial position')}
                </div>
              )}
            </>
          ) : (
            <div className="projection-note">
              {lang === 'he' ? 'רפרנס של משימה 4 לא זמין' : 'Task 4 reference data is not available'}
            </div>
          )}
        </div>
      </div>

      <div className="panel-section">
        <TaskSectionHeader lang={lang} taskKey="projection" />
        <h3>{lang === 'he' ? 'הקרנת פיקסל ל-GPS' : 'Pixel-GPS Projection'}</h3>
        <div className="projection-stack">
          <div className="projection-chips">
            <button
              type="button"
              className={`projection-chip${projectionTarget === 'mission' ? ' projection-chip-active' : ''}`}
              onClick={() => setProjectionTarget('mission')}
            >
              {lang === 'he' ? 'משימת CSV' : 'CSV Mission'}
            </button>
            <button
              type="button"
              className={`projection-chip${projectionTarget === 'image' ? ' projection-chip-active' : ''}`}
              onClick={() => setProjectionTarget('image')}
            >
              {lang === 'he' ? 'משימת תמונות' : 'Image Mission'}
            </button>
          </div>
          <div className="projection-note">
            {activeProjectionMission || (lang === 'he' ? 'לא נבחרה משימה' : 'No mission selected')}
          </div>
          {activeProjectionMeta && (
            <div className="projection-note">
              <div>{lang === 'he' ? 'קנבס' : 'Canvas'}: {activeProjectionMeta.canvas_width_px} x {activeProjectionMeta.canvas_height_px}</div>
              <div>GSD: {activeProjectionMeta.gsd_m} m/px</div>
            </div>
          )}
          <label className="projection-label">
            {lang === 'he' ? 'רוחב תמונת מקור (px)' : 'Source image width (px)'}
          </label>
          <input
            type="number"
            min="1"
            value={projectionForm.imageWidthPx}
            onChange={(e) => handleProjectionField('imageWidthPx', e.target.value)}
            className="projection-input"
          />
          <div className="projection-chips">
            <button type="button" className="projection-chip" onClick={() => fillProjectionPreset('tl')}>TL</button>
            <button type="button" className="projection-chip" onClick={() => fillProjectionPreset('center')}>
              {lang === 'he' ? 'מרכז' : 'Center'}
            </button>
            <button type="button" className="projection-chip" onClick={() => fillProjectionPreset('br')}>BR</button>
          </div>
          <div className="projection-grid">
            <div>
              <label className="projection-label">X</label>
              <input
                type="number"
                value={projectionForm.x}
                onChange={(e) => handleProjectionField('x', e.target.value)}
                className="projection-input"
              />
            </div>
            <div>
              <label className="projection-label">Y</label>
              <input
                type="number"
                value={projectionForm.y}
                onChange={(e) => handleProjectionField('y', e.target.value)}
                className="projection-input"
              />
            </div>
          </div>
          <button type="button" className="projection-btn" onClick={handlePixelToGps}>
            {lang === 'he' ? 'פיקסל → GPS' : 'Pixel → GPS'}
          </button>
          <div className="projection-grid">
            <div>
              <label className="projection-label">Lat</label>
              <input
                type="number"
                step="any"
                value={projectionForm.lat}
                onChange={(e) => handleProjectionField('lat', e.target.value)}
                className="projection-input"
              />
            </div>
            <div>
              <label className="projection-label">Lon</label>
              <input
                type="number"
                step="any"
                value={projectionForm.lon}
                onChange={(e) => handleProjectionField('lon', e.target.value)}
                className="projection-input"
              />
            </div>
          </div>
          <button type="button" className="projection-btn projection-btn-secondary" onClick={handleGpsToPixel}>
            {lang === 'he' ? 'GPS → פיקסל' : 'GPS → Pixel'}
          </button>
          {projectionError && <div className="projection-error">{projectionError}</div>}
          {projectionResult && (
            <div className="projection-result">
              <div><strong>{projectionResult.mode === 'pixel_to_gps' ? 'GPS' : 'Pixel'}</strong></div>
              <div className="projection-result-tag">{getProjectionActionLabel(projectionResult)}</div>
              <div>X: {projectionResult.data.x}</div>
              <div>Y: {projectionResult.data.y}</div>
              <div>Lat: {Number(projectionResult.data.lat).toFixed(6)}</div>
              <div>Lon: {Number(projectionResult.data.lon).toFixed(6)}</div>
            </div>
          )}
        </div>
      </div>

      <div className="panel-section">
        <h3>{t('pnlLayers')}</h3>
        <div className="toggle-list">
          {layerDefs.map(({ key, label, id }) => (
            <div className="toggle-row" key={key}>
              <span>{label}</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  id={id}
                  checked={layers[key]}
                  onChange={(e) => onLayerToggle(key, e.target.checked)}
                />
                <span className="slider" />
              </label>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-section">
        <h3>{t('pnlAlerts')}</h3>
        <div className="alert-list">
          {fps.length === 0 ? (
            <span className="no-alerts">{t('noAlertsMsg')}</span>
          ) : alerts.length === 0 ? (
            <span className="no-alerts">{t('noAlertsOk')}</span>
          ) : (
            alerts.map((a, i) => (
              <div key={i} className={`alert-item ${a.cls}`}>
                <span className="alert-icon">{a.icon}</span>
                {a.text}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="panel-section" style={{ flex: 1 }}>
        <h3>{t('pnlTelemetry')}</h3>
        <div className="waypoint-list">
          {fps.map((fp, i) => {
            const bat = parseFloat(fp.battery_pct);
            const batCls = bat < 40 ? 'badge-danger' : bat < 70 ? 'badge-warn' : 'badge-ok';
            const memCls = fp.memory_integrity !== 'OK' ? 'badge-warn' : 'badge-ok';
            return (
              <div
                key={i}
                id={`wp-${i}`}
                className={`waypoint-item${activeWaypoint === i ? ' active' : ''}`}
                onClick={() => onWaypointClick(fp.nadir.lat, fp.nadir.lon, i)}
              >
                <div className="wp-top">
                  <span className="wp-idx">#{i + 1}</span>
                  <span className="wp-type">{fp.mission_type}</span>
                </div>
                <div className="wp-coords" style={{ direction: 'ltr', textAlign: lang === 'he' ? 'right' : 'left' }}>
                  {fp.nadir.lat.toFixed(5)}, {fp.nadir.lon.toFixed(5)}
                </div>
                <div className="wp-badges">
                  <span className="badge badge-weather">{fp.weather}</span>
                  <span className={`badge ${batCls}`}>{bat.toFixed(0)}%</span>
                  <span className={`badge ${memCls}`}>{fp.memory_integrity !== 'OK' ? 'WARN' : 'OK'}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {isImagePreviewModalOpen && imagePreviewUrl && (
        <div className="image-preview-modal" onClick={() => setIsImagePreviewModalOpen(false)}>
          <div className="image-preview-modal-card" onClick={(event) => event.stopPropagation()}>
            <div className="image-preview-modal-head">
              <strong>{lang === 'he' ? 'תצוגה גדולה - מוזאיקה' : 'Large Mosaic Preview'}</strong>
              <button
                type="button"
                className="image-preview-modal-close"
                onClick={() => setIsImagePreviewModalOpen(false)}
              >
                {lang === 'he' ? 'סגור' : 'Close'}
              </button>
            </div>
            <div className="image-preview-modal-body">
              <img
                src={imagePreviewUrl}
                alt="Large mosaic preview"
                className="image-preview-modal-image"
                onClick={handleImagePreviewClick}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
