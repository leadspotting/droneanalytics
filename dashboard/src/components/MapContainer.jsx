import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { escapeHtml } from '../utils/html';
import 'leaflet-draw';
import 'leaflet-draw/dist/leaflet.draw.css';

// ── Planned-route rendering thresholds ──────────────────────────────────────
// <= FULL_EDIT_THRESHOLD    : every waypoint rendered as a draggable marker
// <= REDUCED_EDIT_THRESHOLD : polyline + every Nth waypoint draggable (sparse)
// >  REDUCED_EDIT_THRESHOLD : polyline + start/end only, drag disabled
const FULL_EDIT_THRESHOLD    = 200;
const REDUCED_EDIT_THRESHOLD = 500;

/**
 * MapContainer — vanilla Leaflet map, initialized once, updated via effects.
 *
 * Props:
 *  fpData         – footprints API response (or null)
 *  hmData         – heatmap API response (or null)
 *  layers         – { path, footprints, heatmap, markers } booleans
 *  lang           – 'en' | 'he'
 *  t              – translator function
 *  activeWaypoint – number (index)
 *  onWaypointClick– (lat, lon, idx) => void
 *  onPolygonDrawn – callback when user draws a polygon for mission planning
 *  plannedMission – mission planning API response (or null)
 *  onWaypointDragged - callback when user drags a planned waypoint
 *  mapFocusPoint  – { lat, lon, x, y, missionId, target } or null
 */
export default function MapContainer({
  fpData,
  hmData,
  layers,
  lang,
  t,
  mapStyle,
  activeWaypoint,
  onWaypointClick,
  onPolygonDrawn,
  plannedMission,
  onWaypointDragged,
  mapFocusPoint,
}) {
  const mapRef     = useRef(null); // Leaflet map instance
  const mapDivRef  = useRef(null); // DOM div
  const layersRef  = useRef(null); // Leaflet layer groups
  const drawLayerRef = useRef(null); // Layer for drawn polygons
  const planLayerRef = useRef(null); // Layer for planned mission waypoints
  const focusLayerRef = useRef(null); // Layer for Task 4 selected map point
  const baseTileLayerRef = useRef(null); // active basemap layer
  const onPolygonDrawnRef = useRef(onPolygonDrawn);
  useEffect(() => {
    onPolygonDrawnRef.current = onPolygonDrawn;
  }, [onPolygonDrawn]);

  // Hint shown in the map when the route is too large for full interactive editing
  const [routeHint, setRouteHint] = useState(null);

  // ── Initialize map once ─────────────────────────────────────────
  useEffect(() => {
    if (mapRef.current) return; // already initialized

    const map = L.map(mapDivRef.current, {
      center: [32.08, 34.78],
      zoom: 11,
      zoomControl: true,
      attributionControl: true,
    });

    const lg = {
      path:       L.layerGroup().addTo(map),
      footprints: L.layerGroup().addTo(map),
      heatmap:    L.layerGroup().addTo(map),
      markers:    L.layerGroup().addTo(map),
    };
    
    // Add draw layer
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    drawLayerRef.current = drawnItems;
    
    // Add planned mission layer
    const planLayer = L.layerGroup().addTo(map);
    planLayerRef.current = planLayer;

    const focusLayer = L.layerGroup().addTo(map);
    focusLayerRef.current = focusLayer;

    // Initialize Leaflet Draw Control
    const drawControl = new L.Control.Draw({
      draw: {
        polyline: false,
        marker: false,
        circlemarker: false,
        circle: false,
        polygon: true,
        rectangle: false,
      },
      edit: {
        featureGroup: drawnItems,
        remove: true
      }
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, function (e) {
      drawnItems.clearLayers(); // Keep only one polygon at a time
      const layer = e.layer;
      drawnItems.addLayer(layer);
      
      // Extract coordinates and send to parent
      let latlngs = [];
      if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
        // Get the lat/lons of the polygon vertices
        latlngs = layer.getLatLngs()[0].map(pt => ({ lat: pt.lat, lon: pt.lng }));
      }
      
      if (onPolygonDrawnRef.current && latlngs.length >= 3) {
        onPolygonDrawnRef.current(latlngs);
      }
    });
    
    // Listen for deleted / edited items to clear the planned mission if needed
    map.on(L.Draw.Event.DELETED, function (e) {
        onPolygonDrawnRef.current?.(null); // Clear selection
    });
    map.on(L.Draw.Event.EDITED, () => {
      const polygon = drawnItems.getLayers()[0];
      if (polygon) {
        onPolygonDrawnRef.current?.(polygon.getLatLngs()[0].map(pt => ({ lat: pt.lat, lon: pt.lng })));
      }
    });

    mapRef.current    = map;
    layersRef.current = lg;
    const resizeObserver = new ResizeObserver(() => map.invalidateSize({ pan: false }));
    resizeObserver.observe(mapDivRef.current);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current    = null;
      layersRef.current = null;
      drawLayerRef.current = null;
      planLayerRef.current = null;
      focusLayerRef.current = null;
      baseTileLayerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const nextBaseLayer = createBaseTileLayer(mapStyle);
    if (baseTileLayerRef.current) {
      map.removeLayer(baseTileLayerRef.current);
    }
    nextBaseLayer.addTo(map);
    baseTileLayerRef.current = nextBaseLayer;
  }, [mapStyle]);

  // ── Re-render mission data when fpData / hmData / lang change ───
  useEffect(() => {
    const map = mapRef.current;
    const lg  = layersRef.current;
    if (!map || !lg) return;

    // Clear all layers
    Object.values(lg).forEach(l => l.clearLayers());

    const fps = fpData?.footprints;
    if (!fps || fps.length === 0) return;

    // Flight path polyline
    const latlngs = fps.map(f => [f.nadir.lat, f.nadir.lon]);
    const path = L.polyline(latlngs, {
      color: '#3b82f6', weight: 2.5, opacity: 0.85, dashArray: '6 4',
    });
    lg.path.addLayer(path);

    // Footprints + markers
    fps.forEach((fp, i) => {
      // Camera footprint polygon
      const poly = L.polygon(fp.corners.map(c => [c.lat, c.lon]), {
        color: '#06b6d4', weight: 1,
        fillColor: '#06b6d4', fillOpacity: 0.06, opacity: 0.5,
      });
      poly.bindPopup(makePopup(fp, i + 1, t));
      lg.footprints.addLayer(poly);

      // Waypoint circle marker
      const isFirst = i === 0, isLast = i === fps.length - 1;
      const color   = isFirst ? '#22c55e' : isLast ? '#ef4444' : '#3b82f6';
      const marker  = L.circleMarker([fp.nadir.lat, fp.nadir.lon], {
        radius: isFirst || isLast ? 9 : 5,
        color, fillColor: color, fillOpacity: 0.9, weight: 2,
      });
      marker.bindPopup(makePopup(fp, i + 1, t));
      marker.on('click', () => onWaypointClick(fp.nadir.lat, fp.nadir.lon, i));
      lg.markers.addLayer(marker);
    });

    // Start / End labels
    addLabel(lg.markers, fps[0].nadir, t('pStart'), '#22c55e');
    addLabel(lg.markers, fps[fps.length - 1].nadir, t('pEnd'), '#ef4444');

    // Heatmap cells
    if (hmData?.cells) {
      hmData.cells.forEach(cell => {
        const col = cell.count >= 3 ? '#ef4444' : cell.count === 2 ? '#eab308' : '#22c55e';
        const delta = 0.0014;
        const rect = L.rectangle(
          [[cell.lat - delta, cell.lon - delta], [cell.lat + delta, cell.lon + delta]],
          { color: col, weight: 0, fillColor: col, fillOpacity: 0.32 }
        );
        rect.bindTooltip(`${t('pCoverageCount')}: ${cell.count}x`, { sticky: true });
        lg.heatmap.addLayer(rect);
      });
    }

    // Fit bounds
    try { map.fitBounds(path.getBounds(), { padding: [30, 30] }); } catch (_) {}

  }, [fpData, hmData, lang, t, onWaypointClick]);

  // ── Toggle layer visibility ──────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    const lg  = layersRef.current;
    if (!map || !lg) return;

    const toggle = (name, show) => {
      if (show) { if (!map.hasLayer(lg[name])) lg[name].addTo(map); }
      else       { if (map.hasLayer(lg[name])) map.removeLayer(lg[name]); }
    };

    toggle('path',       layers.path);
    toggle('footprints', layers.footprints);
    toggle('heatmap',    layers.heatmap);
    toggle('markers',    layers.markers);
  }, [layers]);

  // ── Fly to active waypoint when changed via sidebar ─────────────
  useEffect(() => {
    const map = mapRef.current;
    const lg  = layersRef.current;
    if (!map || !lg || activeWaypoint < 0) return;

    const fps = fpData?.footprints;
    if (!fps || !fps[activeWaypoint]) return;

    const { lat, lon } = fps[activeWaypoint].nadir;
    map.flyTo([lat, lon], 15, { duration: 0.8 });
  }, [activeWaypoint, fpData]);
  
  // ── Render planned mission on map ───────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    const planLayer = planLayerRef.current;
    if (!map || !planLayer) return;

    planLayer.clearLayers();

    if (!plannedMission?.waypoints?.length) {
      setRouteHint(null);
      return;
    }

    const wps     = plannedMission.waypoints;
    const count   = wps.length;
    const latlngs = wps.map(wp => [wp.lat, wp.lon]);

    // Route polyline — always rendered regardless of mode
    const path = L.polyline(latlngs, {
      color: '#f59e0b', weight: 2, opacity: 0.9, dashArray: '4 4',
    });
    planLayer.addLayer(path);

    if (count <= FULL_EDIT_THRESHOLD) {
      // Full mode: every waypoint rendered as a draggable marker
      setRouteHint(null);
      latlngs.forEach((coord, idx) => {
        planLayer.addLayer(makeWpMarker(coord, idx, count, true, onWaypointDragged));
      });
    } else if (count <= REDUCED_EDIT_THRESHOLD) {
      // Sparse mode: start/end + every Nth waypoint, all draggable
      const step = Math.ceil(count / FULL_EDIT_THRESHOLD);
      setRouteHint(
        `Large route (${count} waypoints) — showing every ${step}th marker. Drag enabled on visible markers only.`
      );
      latlngs.forEach((coord, idx) => {
        if (idx === 0 || idx === count - 1 || idx % step === 0) {
          planLayer.addLayer(makeWpMarker(coord, idx, count, true, onWaypointDragged));
        }
      });
    } else {
      // Path-only mode: start/end endpoints only, drag disabled
      setRouteHint(
        `Very large route (${count} waypoints) — interactive editing disabled. Export the route to modify it.`
      );
      planLayer.addLayer(makeWpMarker(latlngs[0],         0,         count, false, null));
      planLayer.addLayer(makeWpMarker(latlngs[count - 1], count - 1, count, false, null));
    }

    try { map.fitBounds(path.getBounds(), { padding: [20, 20] }); } catch (_) {}
  }, [plannedMission, onWaypointDragged]);

  useEffect(() => {
    const map = mapRef.current;
    const focusLayer = focusLayerRef.current;
    if (!map || !focusLayer) return;

    focusLayer.clearLayers();
    if (!mapFocusPoint) return;

    const pos = [mapFocusPoint.lat, mapFocusPoint.lon];
    const ring = L.circleMarker(pos, {
      radius: 10,
      color: '#f8fafc',
      weight: 3,
      fillColor: '#06b6d4',
      fillOpacity: 0.95,
    });
    const halo = L.circleMarker(pos, {
      radius: 18,
      color: '#06b6d4',
      weight: 2,
      fillOpacity: 0,
      opacity: 0.45,
    });
    const label = L.marker(pos, {
      icon: L.divIcon({
        className: 'task4-focus-label',
        html: `
          <div class="task4-focus-pill">
            <strong>${escapeHtml(mapFocusPoint.missionId)}</strong><br/>
            x=${mapFocusPoint.x}, y=${mapFocusPoint.y}<br/>
            ${mapFocusPoint.lat.toFixed(6)}, ${mapFocusPoint.lon.toFixed(6)}
          </div>
        `,
        iconAnchor: [0, 34],
      }),
    });

    ring.bindPopup(`
      <div class="popup-title">Task 4 Selection</div>
      <b>Mission</b> ${escapeHtml(mapFocusPoint.missionId)}<br>
      <b>Pixel</b> ${mapFocusPoint.x}, ${mapFocusPoint.y}<br>
      <b>Lat</b> ${mapFocusPoint.lat.toFixed(6)}<br>
      <b>Lon</b> ${mapFocusPoint.lon.toFixed(6)}
    `);

    focusLayer.addLayer(halo);
    focusLayer.addLayer(ring);
    focusLayer.addLayer(label);
    map.flyTo(pos, Math.max(map.getZoom(), 16), { duration: 0.8 });
  }, [mapFocusPoint]);

  return (
    <div id="map" style={{ position: 'relative' }}>
      <div ref={mapDivRef} style={{ width: '100%', height: '100%' }} />
      {routeHint && (
        <div style={{
          position: 'absolute',
          bottom: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(15, 20, 40, 0.92)',
          color: '#f59e0b',
          border: '1px solid rgba(245, 158, 11, 0.35)',
          borderRadius: 8,
          padding: '6px 16px',
          fontSize: 12,
          fontWeight: 500,
          pointerEvents: 'none',
          zIndex: 1000,
          whiteSpace: 'nowrap',
          maxWidth: '90%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          {routeHint}
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────

/**
 * Build a single planned-waypoint marker.
 * @param {[number, number]} coord     - [lat, lon]
 * @param {number}           idx       - waypoint index (0-based)
 * @param {number}           total     - total waypoint count
 * @param {boolean}          draggable - whether the marker can be dragged
 * @param {Function|null}    onDragged - (idx, lat, lon) callback; ignored when draggable=false
 */
function makeWpMarker(coord, idx, total, draggable, onDragged) {
  const isFirst = idx === 0, isLast = idx === total - 1;
  const color   = isFirst ? '#22c55e' : isLast ? '#ef4444' : '#f59e0b';

  const icon = L.divIcon({
    className: 'custom-wp-drag-icon',
    html: `<div style="width:14px;height:14px;background-color:${color};border-radius:50%;border:2px solid white;box-shadow:0 0 3px rgba(0,0,0,0.5);"></div>`,
    iconSize:     [14, 14],
    iconAnchor:   [7, 7],
    tooltipAnchor: [10, 0],
  });

  const marker = L.marker(coord, { icon, draggable });
  marker.bindTooltip(`Planned WP #${idx + 1}`);

  if (draggable && onDragged) {
    marker.on('dragend', (e) => {
      const { lat, lng } = e.target.getLatLng();
      onDragged(idx, lat, lng);
    });
  }

  return marker;
}

function createBaseTileLayer(mapStyle) {
  const tileConfigByStyle = {
    light: {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      options: {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      },
    },
    dark: {
      url: 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png',
      options: {
        maxZoom: 20,
        attribution: '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'
      },
    },
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      options: { maxZoom: 20 },
    },
    terrain: {
      url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
      options: { maxZoom: 17 },
    },
  };

  const selectedConfig = tileConfigByStyle[mapStyle] ?? tileConfigByStyle.dark;
  return L.tileLayer(selectedConfig.url, selectedConfig.options);
}

function addLabel(layerGroup, pos, text, color) {
  const icon = L.divIcon({
    html: `<div style="background:${color}20;border:1px solid ${color};color:${color};padding:2px 7px;border-radius:6px;font-family:Inter,sans-serif;font-size:11px;font-weight:600;white-space:nowrap;">${text}</div>`,
    className: '',
    iconAnchor: [0, 0],
  });
  layerGroup.addLayer(L.marker([pos.lat, pos.lon], { icon }));
}

function makePopup(fp, idx, t) {
  const tamper = fp.gps_tampering === '1' ? t('pTamper') : '';
  const leak   = fp.data_leak     === '1' ? t('pLeak')   : '';
  const mem    = fp.memory_integrity !== 'OK' ? `${t('pMem')} ${escapeHtml(fp.memory_integrity)}` : '';
  const alerts = [tamper, leak, mem].filter(Boolean).join('<br>') || t('pAllClear');

  return `
    <div class="popup-title">📍 WP #${idx}</div>
    <b>${t('pTime')}</b> ${escapeHtml(fp.timestamp)}<br>
    <b>${t('pType')}</b> ${escapeHtml(fp.mission_type)}<br>
    <b>${t('pAlt')}</b>  ${fp.altitude_m} m<br>
    <b>${t('pFoot')}</b> ${fp.width_m} m × ${fp.height_m} m<br>
    <b>${t('pBat')}</b>  ${parseFloat(fp.battery_pct).toFixed(1)}%<br>
    <b>${t('pSig')}</b>  ${parseFloat(fp.signal_pct).toFixed(1)}%<br>
    <b>${t('pWeather')}</b> ${escapeHtml(fp.weather)} ${escapeHtml(fp.wind_m_s)} ${t('windLbl')}<br>
    <b>${t('pSec')}</b>  ${alerts}
  `;
}
