import { useCallback, useEffect, useRef, useState } from 'react';
import Header from './components/Header';
import MapContainer from './components/MapContainer';
import Sidebar from './components/Sidebar/Sidebar';
import { useLanguage } from './hooks/useLanguage';
import { useTheme } from './hooks/useTheme';
import {
  createCustomDrone,
  fetchDrones,
  fetchImageMissions,
  fetchImageMosaic,
  fetchMissionData,
  fetchMissions,
  fetchTask4Reference,
  planMission,
  runRenderJob,
  recommendFlightParams as recommendFlightParamsRequest,
  projectGpsToPixel as projectGpsToPixelRequest,
  projectPixelToGps as projectPixelToGpsRequest,
  uploadTask4Reference as uploadTask4ReferenceRequest,
  uploadImageMissionFiles as uploadImageMissionFilesRequest,
  uploadMissionCsv,
} from './services/api';
import { buildImageExportJob, buildImagePreviewJob, sleep } from './utils/imageMission';
import { computeFeasibility, computeFlightTime } from './utils/missionMath';

const DEFAULT_LAYER_VISIBILITY = {
  path: true,
  footprints: true,
  heatmap: true,
  markers: true,
};

export default function App() {
  const { lang, t, toggleLang } = useLanguage();
  const { theme, toggleTheme } = useTheme();

  const [missions, setMissions] = useState([]);
  const [selectedMission, setSelectedMission] = useState('');
  const [missionRevision, setMissionRevision] = useState(0);
  const [fpData, setFpData] = useState(null);
  const [hmData, setHmData] = useState(null);
  const [mosaicMeta, setMosaicMeta] = useState(null);

  const [imageMissions, setImageMissions] = useState([]);
  const [selectedImageMission, setSelectedImageMission] = useState('');
  const [imageRevision, setImageRevision] = useState(0);
  const [imageMosaicMeta, setImageMosaicMeta] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [imagePreviewScale, setImagePreviewScale] = useState(0.12);
  const [imagePreviewMode, setImagePreviewMode] = useState('single');
  const [imagePreviewJob, setImagePreviewJob] = useState(null);
  const previewAbortRef = useRef(null);

  const [task4ReferenceMeta, setTask4ReferenceMeta] = useState(null);
  const [task4ReferencePreviewUrl, setTask4ReferencePreviewUrl] = useState('');
  const [mapFocusPoint, setMapFocusPoint] = useState(null);

  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [activeWaypoint, setActiveWaypoint] = useState(-1);
  const [layers, setLayers] = useState(DEFAULT_LAYER_VISIBILITY);
  const [mapStyle, setMapStyle] = useState('dark');

  const [savedMission, setSavedMission] = useState(null);
  const [draftMission, setDraftMission] = useState(null);
  const [editHistory, setEditHistory] = useState([]);

  const [drones, setDrones] = useState([]);
  const [selectedDrone, setSelectedDrone] = useState(null);
  const [savedDroneId, setSavedDroneId] = useState(null);
  const [feasibility, setFeasibility] = useState(null);

  const refreshFeasibility = useCallback((waypoints, drone) => {
    setFeasibility(computeFeasibility(waypoints, drone));
  }, []);

  const refreshImageMissions = useCallback(async ({ retries = 1, delayMs = 0 } = {}) => {
    for (let attempt = 0; attempt < retries; attempt += 1) {
      try {
        const data = await fetchImageMissions();
        const missionsData = data ?? [];
        setImageMissions(missionsData);

        setSelectedImageMission((currentMission) => (
          currentMission && missionsData.some((mission) => mission.mission_id === currentMission)
            ? currentMission
            : (missionsData[0]?.mission_id ?? '')
        ));

        return missionsData;
      } catch {
        // Silent retry during bootstrap to keep the initial UX clean.
      }

      if (attempt < retries - 1 && delayMs > 0) {
        await sleep(delayMs);
      }
    }

    setImageMissions([]);
    return [];
  }, []);

  const loadTask4ReferenceData = useCallback(async () => {
    try {
      const referenceData = await fetchTask4Reference();
      if (!referenceData) {
        setTask4ReferenceMeta(null);
        setTask4ReferencePreviewUrl('');
        return;
      }

      setTask4ReferenceMeta(referenceData);
      setTask4ReferencePreviewUrl(`${referenceData.preview_url}&t=${Date.now()}`);
    } catch {
      setTask4ReferenceMeta(null);
      setTask4ReferencePreviewUrl('');
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    async function initializeDashboard() {
      // Vite can start before Python has finished loading. Retry until the API
      // is ready instead of leaving the map empty until the page is refreshed.
      for (let attempt = 0; attempt < 20 && !controller.signal.aborted; attempt += 1) {
        try {
          const options = { signal: controller.signal };
          const [missionsData, dronesData] = await Promise.all([
            fetchMissions(options),
            fetchDrones(options),
          ]);
          if (controller.signal.aborted) return;
          setMissions(missionsData);
          setDrones(dronesData.drones ?? []);
          setSelectedMission(missionsData[0]?.mission_id ?? '');
          setUploadStatus('');
          break;
        } catch {
          if (controller.signal.aborted) return;
          if (attempt === 19) setUploadStatus('Could not connect to the backend. Check that the server is running.');
          else await sleep(1000);
        }
      }

      if (controller.signal.aborted) return;
      await refreshImageMissions({ retries: 20, delayMs: 1000 });
      if (controller.signal.aborted) return;
      await loadTask4ReferenceData();
    }

    initializeDashboard();
    return () => controller.abort();
  }, [loadTask4ReferenceData, refreshImageMissions]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadMission(missionId) {
      if (!missionId) {
        setFpData(null);
        setHmData(null);
        setMosaicMeta(null);
        setLoading(false);
        return;
      }

      setLoading(true);
      setActiveWaypoint(-1);
      try {
        const missionData = await fetchMissionData(missionId, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setFpData(missionData.footprints);
        setHmData(missionData.heatmap);
        setMosaicMeta(missionData.mosaic);
      } catch {
        if (controller.signal.aborted) return;
        setFpData(null);
        setHmData(null);
        setMosaicMeta(null);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    loadMission(selectedMission);
    return () => controller.abort();
  }, [selectedMission, missionRevision]);

  const cancelImagePreview = useCallback(() => {
    previewAbortRef.current?.abort();
    previewAbortRef.current = null;
    setImagePreviewJob(null);
  }, []);

  // Render the mosaic preview as a background job; progress is shown in the sidebar.
  const requestImagePreview = useCallback(async (missionId, scale, mode, force = false) => {
    previewAbortRef.current?.abort();
    const controller = new AbortController();
    previewAbortRef.current = controller;
    setImagePreviewJob({ status: 'queued', progress: 0, stage: '' });
    try {
      const resultUrl = await runRenderJob(
        buildImagePreviewJob(missionId, scale, mode, force),
        { signal: controller.signal, onProgress: (status) => { if (!controller.signal.aborted) setImagePreviewJob(status); } },
      );
      if (controller.signal.aborted) return;
      setImagePreviewUrl(`${resultUrl}?t=${Date.now()}`);
      setImagePreviewJob(null);
    } catch (error) {
      if (controller.signal.aborted) return;
      setImagePreviewJob({ status: 'error', progress: 0, error: error.message ?? 'Render failed' });
    }
  }, []);

  useEffect(() => () => previewAbortRef.current?.abort(), []);

  useEffect(() => {
    let active = true;
    async function loadSelectedImageMission() {
      if (!selectedImageMission) {
        setImageMosaicMeta(null);
        setImagePreviewUrl('');
        cancelImagePreview();
        return;
      }

      try {
        const mosaicData = await fetchImageMosaic(selectedImageMission);
        if (!active) return;
        const previewScale = 0.12;

        setImageMosaicMeta(mosaicData);
        setImagePreviewScale(previewScale);
        requestImagePreview(selectedImageMission, previewScale, imagePreviewMode);
      } catch {
        if (!active) return;
        setImageMosaicMeta(null);
        setImagePreviewUrl('');
      }
    }

    loadSelectedImageMission();
    return () => { active = false; };
  }, [imagePreviewMode, selectedImageMission, imageRevision, requestImagePreview, cancelImagePreview]);

  async function loadDefaultImageMission() {
    setLoading(true);
    try {
      const missionsData = await refreshImageMissions();
      if (!missionsData.length) {
        throw new Error('No image missions found. Upload images first.');
      }
      setSelectedImageMission(missionsData[0].mission_id);
    } catch (error) {
      setUploadStatus(error.message ?? 'Failed to load image mission');
    } finally {
      setLoading(false);
    }
  }

  async function handleImageMissionUpload({ files, missionId, aglM, onProgress }) {
    const missionData = await uploadImageMissionFilesRequest({ files, missionId, aglM, onProgress });
    await refreshImageMissions();
    const mid = missionData.mission_id;
    setSelectedImageMission(mid);
    setImageRevision((revision) => revision + 1);
    // Force the preview URL to rebuild even if the mission_id is unchanged
    // (the useEffect on selectedImageMission won't fire when the value is the same)
    requestImagePreview(mid, imagePreviewScale, imagePreviewMode);
    return missionData;
  }

  function handleReRender() {
    if (!selectedImageMission) return;
    requestImagePreview(selectedImageMission, imagePreviewScale, imagePreviewMode, true);
  }

  async function projectPixelToGps({ x, y, imageWidthPx, target = 'mission', missionId }) {
    const activeMissionId = missionId ?? (target === 'image' ? selectedImageMission : selectedMission);
    if (!activeMissionId) throw new Error('No mission selected');

    return projectPixelToGpsRequest({
      missionId: activeMissionId,
      target,
      x,
      y,
      imageWidthPx,
    });
  }

  async function projectGpsToPixel({ lat, lon, imageWidthPx, target = 'mission', missionId }) {
    const activeMissionId = missionId ?? (target === 'image' ? selectedImageMission : selectedMission);
    if (!activeMissionId) throw new Error('No mission selected');

    return projectGpsToPixelRequest({
      missionId: activeMissionId,
      target,
      lat,
      lon,
      imageWidthPx,
    });
  }

  async function handleImagePreviewPick({ x, y, imageWidthPx, target = 'image', missionId }) {
    const projectionData = await projectPixelToGps({ x, y, imageWidthPx, target, missionId });
    setMapFocusPoint({
      lat: projectionData.lat,
      lon: projectionData.lon,
      x: projectionData.x,
      y: projectionData.y,
      missionId: projectionData.mission_id,
      target,
    });
    return projectionData;
  }

  async function handleTask4PreviewPick({ x, y, xNorm, yNorm }) {
    if (!task4ReferenceMeta) {
      throw new Error('Task 4 reference is not available');
    }

    let lat;
    let lon;

    if (Array.isArray(task4ReferenceMeta.corners) && task4ReferenceMeta.corners.length === 4) {
      const [topLeft, topRight, bottomRight, bottomLeft] = task4ReferenceMeta.corners;
      const topLat = topLeft.lat * (1 - xNorm) + topRight.lat * xNorm;
      const topLon = topLeft.lon * (1 - xNorm) + topRight.lon * xNorm;
      const bottomLat = bottomLeft.lat * (1 - xNorm) + bottomRight.lat * xNorm;
      const bottomLon = bottomLeft.lon * (1 - xNorm) + bottomRight.lon * xNorm;
      lat = topLat * (1 - yNorm) + bottomLat * yNorm;
      lon = topLon * (1 - yNorm) + bottomLon * yNorm;
    } else if (task4ReferenceMeta.bounding_box) {
      const { west, east, north, south } = task4ReferenceMeta.bounding_box;
      lon = west + (east - west) * xNorm;
      lat = north - (north - south) * yNorm;
    } else {
      throw new Error('Task 4 reference is missing geospatial metadata');
    }

    const projectionData = {
      mission_id: 'task4_reference',
      target: 'task4_reference',
      x,
      y,
      lat,
      lon,
    };

    setMapFocusPoint(projectionData);
    return projectionData;
  }

  async function handleRecommendFlightParams(payload) {
    return recommendFlightParamsRequest(payload);
  }

  async function handleTask4ReferenceUpload(file) {
    const data = await uploadTask4ReferenceRequest(file);
    setTask4ReferenceMeta(data);
    setTask4ReferencePreviewUrl(`${data.preview_url}&t=${Date.now()}`);
    return data;
  }

  const handleLayerToggle = useCallback((layerName, checked) => {
    setLayers((previousLayers) => ({ ...previousLayers, [layerName]: checked }));
  }, []);

  const handleWaypointClick = useCallback((_lat, _lon, index) => {
    setActiveWaypoint(index);
    window.setTimeout(() => {
      document.getElementById(`wp-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 50);
  }, []);

  async function handleUpload(file) {
    setLoading(true);
    try {
      const uploadData = await uploadMissionCsv(file);
      const missionsData = await fetchMissions();
      setUploadStatus(`${t('uploadSucceed')} (${uploadData.total_rows})`);
      setMissions(missionsData);
      setSelectedMission(missionsData[0]?.mission_id ?? '');
      setMissionRevision((revision) => revision + 1);
    } catch {
      setUploadStatus(t('uploadFail'));
    } finally {
      setLoading(false);
    }
  }

  const handleDroneSelect = useCallback((droneId) => {
    const drone = drones.find((candidate) => candidate.id === droneId) ?? null;
    setSelectedDrone(drone);
    if (draftMission?.waypoints) {
      setFeasibility(computeFeasibility(draftMission.waypoints, drone));
    }
  }, [draftMission, drones]);

  const handleAddCustomDrone = useCallback(async (formData) => {
    const data = await createCustomDrone(formData);
    setDrones(data.drones);

    const newDrone = data.drone;
    setSelectedDrone(newDrone);
    if (draftMission?.waypoints) {
      setFeasibility(computeFeasibility(draftMission.waypoints, newDrone));
    }
  }, [draftMission]);

  const handlePolygonDrawn = useCallback(async (polygonPoints) => {
    if (!polygonPoints) {
      setSavedMission(null);
      setDraftMission(null);
      setEditHistory([]);
      setSavedDroneId(null);
      setFeasibility(null);
      return;
    }

    try {
      setLoading(true);
      const missionData = await planMission(polygonPoints);
      setSavedMission(missionData);
      setDraftMission(missionData);
      setEditHistory([]);
      setSavedDroneId(null);
      setFeasibility(computeFeasibility(missionData.waypoints, selectedDrone));
    } catch {
      setDraftMission(null);
    } finally {
      setLoading(false);
    }
  }, [selectedDrone]);

  const handleWaypointDragged = useCallback((index, lat, lon) => {
    setEditHistory((history) => {
      const nextHistory = [...history, draftMission.waypoints];
      return nextHistory.length > 50 ? nextHistory.slice(-50) : nextHistory;
    });

    setDraftMission((previousMission) => {
      if (!previousMission) return previousMission;

      const updatedWaypoints = previousMission.waypoints.map((waypoint, waypointIndex) => (
        waypointIndex === index ? { ...waypoint, lat, lon } : waypoint
      ));

      refreshFeasibility(updatedWaypoints, selectedDrone);
      return {
        ...previousMission,
        waypoints: updatedWaypoints,
        flight_time_sec: computeFlightTime(updatedWaypoints),
      };
    });
  }, [draftMission, refreshFeasibility, selectedDrone]);

  const handleSaveDraft = useCallback(() => {
    setSavedMission(draftMission);
    setSavedDroneId(selectedDrone?.id ?? null);
    setEditHistory([]);
  }, [draftMission, selectedDrone]);

  const handleUndo = useCallback(() => {
    if (editHistory.length === 0) return;

    const previousWaypoints = editHistory[editHistory.length - 1];
    setDraftMission((currentDraftMission) => {
      if (!currentDraftMission) return currentDraftMission;

      const updatedDraft = {
        ...currentDraftMission,
        waypoints: previousWaypoints,
        flight_time_sec: computeFlightTime(previousWaypoints),
      };
      refreshFeasibility(previousWaypoints, selectedDrone);
      return updatedDraft;
    });
    setEditHistory((history) => history.slice(0, -1));
  }, [editHistory, refreshFeasibility, selectedDrone]);

  const hasUnsavedChanges =
    (draftMission !== null && savedMission !== null && draftMission.waypoints !== savedMission.waypoints)
    || (draftMission !== null && (selectedDrone?.id ?? null) !== savedDroneId);

  const canUndo = editHistory.length > 0;
  const isFeasible = Boolean(selectedDrone) && feasibility?.feasible === true;

  return (
    <>
      <div id="loading" className={loading ? 'show' : ''}>
        <div className="spinner" />
        <span style={{ color: 'var(--muted)', fontSize: 14 }}>{t('loadingText')}</span>
      </div>

      <Header
        lang={lang}
        t={t}
        theme={theme}
        mapStyle={mapStyle}
        onLangToggle={toggleLang}
        onThemeToggle={toggleTheme}
        onMapStyleChange={setMapStyle}
        missions={missions}
        selectedMission={selectedMission}
        onMissionChange={setSelectedMission}
        onUpload={handleUpload}
        uploadStatus={uploadStatus}
      />

      <div className="main">
        <Sidebar
          fpData={fpData}
          hmData={hmData}
          layers={layers}
          onLayerToggle={handleLayerToggle}
          lang={lang}
          t={t}
          activeWaypoint={activeWaypoint}
          onWaypointClick={handleWaypointClick}
          draftMission={draftMission}
          savedMission={savedMission}
          hasUnsavedChanges={hasUnsavedChanges}
          canUndo={canUndo}
          onSaveDraft={handleSaveDraft}
          onUndo={handleUndo}
          drones={drones}
          selectedDrone={selectedDrone}
          onDroneSelect={handleDroneSelect}
          onAddCustomDrone={handleAddCustomDrone}
          feasibility={feasibility}
          isFeasible={isFeasible}
          selectedMission={selectedMission}
          mosaicMeta={mosaicMeta}
          imageMissions={imageMissions}
          selectedImageMission={selectedImageMission}
          onSelectImageMission={setSelectedImageMission}
          imageMosaicMeta={imageMosaicMeta}
          imagePreviewUrl={imagePreviewUrl}
          imagePreviewScale={imagePreviewScale}
          imagePreviewMode={imagePreviewMode}
          onImagePreviewModeChange={setImagePreviewMode}
          imagePreviewJob={imagePreviewJob}
          onCancelImagePreview={cancelImagePreview}
          imageExportJob={selectedImageMission ? buildImageExportJob(selectedImageMission, imagePreviewScale) : null}
          onLoadDefaultImageMission={loadDefaultImageMission}
          onUploadImageMission={handleImageMissionUpload}
          onReRender={handleReRender}
          task4ReferenceMeta={task4ReferenceMeta}
          task4ReferencePreviewUrl={task4ReferencePreviewUrl}
          onImagePreviewPick={handleImagePreviewPick}
          onTask4PreviewPick={handleTask4PreviewPick}
          onTask4ReferenceUpload={handleTask4ReferenceUpload}
          onRecommendFlightParams={handleRecommendFlightParams}
          onProjectPixelToGps={projectPixelToGps}
          onProjectGpsToPixel={projectGpsToPixel}
        />

        <MapContainer
          fpData={fpData}
          hmData={hmData}
          layers={layers}
          lang={lang}
          t={t}
          mapStyle={mapStyle}
          activeWaypoint={activeWaypoint}
          onWaypointClick={handleWaypointClick}
          onPolygonDrawn={handlePolygonDrawn}
          plannedMission={draftMission}
          onWaypointDragged={handleWaypointDragged}
          mapFocusPoint={mapFocusPoint}
        />
      </div>
    </>
  );
}
