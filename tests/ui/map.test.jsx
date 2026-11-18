import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const state = vi.hoisted(() => ({ maps: [], tiles: [], observers: [] }));
vi.mock('leaflet-draw', () => ({}));
vi.mock('leaflet', () => {
  const shape = () => ({ addTo: vi.fn(function () { return this; }), clearLayers: vi.fn(), addLayer: vi.fn(), bindPopup: vi.fn(), bindTooltip: vi.fn(), on: vi.fn(), getBounds: () => [] });
  class Polygon { getLatLngs() { return [[{ lat: 32, lng: 34 }, { lat: 32.01, lng: 34 }, { lat: 32, lng: 34.01 }]]; } }
  return { default: {
    map: vi.fn(() => {
      const map = { handlers: {}, remove: vi.fn(), addLayer: vi.fn(), addControl: vi.fn(), removeLayer: vi.fn(), hasLayer: () => true, fitBounds: vi.fn(), invalidateSize: vi.fn(), on(event, callback) { this.handlers[event] = callback; } };
      state.maps.push(map);
      return map;
    }),
    tileLayer: vi.fn((url, options) => { const tile = { url, options, ...shape() }; state.tiles.push(tile); return tile; }),
    layerGroup: shape, FeatureGroup: function () { return shape(); },
    Control: { Draw: function () {} }, Draw: { Event: { CREATED: 'created', DELETED: 'deleted', EDITED: 'edited' } },
    Polygon, Rectangle: Polygon,
  } };
});
import MapContainer from '../../dashboard/src/components/MapContainer.jsx';

let root, container;
const translate = (key) => key;
const baseProps = { fpData: null, hmData: null, layers: { path: true, footprints: true, heatmap: true, markers: true }, lang: 'en', t: translate, mapStyle: 'dark', activeWaypoint: -1, onWaypointClick: () => {}, onPolygonDrawn: () => {} };
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  state.maps = []; state.tiles = []; state.observers = [];
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback) { this.callback = callback; state.observers.push(this); }
    observe() {} disconnect() {}
  });
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.unstubAllGlobals(); });

test('all map styles switch without destroying the map or overlays', () => {
  act(() => root.render(<MapContainer {...baseProps} />));
  const map = state.maps[0];
  for (const mapStyle of ['light', 'satellite', 'terrain', 'dark']) {
    act(() => root.render(<MapContainer {...baseProps} mapStyle={mapStyle} onPolygonDrawn={() => {}} />));
  }
  expect(state.maps).toHaveLength(1);
  expect(map.remove).not.toHaveBeenCalled();
  expect(state.tiles).toHaveLength(5);
  // Initial dark, then light, satellite, terrain, dark again.
  const expectedHosts = ['stadiamaps.com', 'openstreetmap.org', 'arcgisonline.com', 'opentopomap.org', 'stadiamaps.com'];
  state.tiles.forEach((tile, index) => {
    expect(tile.options.attribution).toBeTruthy();   // every provider requires attribution
    expect(tile.url).toMatch(/^https:\/\//);          // no mixed content / plain http
    expect(tile.url).toContain(expectedHosts[index]);
  });
  expect(map.removeLayer).toHaveBeenCalledTimes(4);
});

test('drawing uses the current callback after drone selection changes', () => {
  const initial = vi.fn(), current = vi.fn();
  act(() => root.render(<MapContainer {...baseProps} onPolygonDrawn={initial} />));
  act(() => root.render(<MapContainer {...baseProps} onPolygonDrawn={current} />));
  act(() => state.maps[0].handlers.deleted());
  expect(initial).not.toHaveBeenCalled();
  expect(current).toHaveBeenCalledWith(null);
  expect(state.maps).toHaveLength(1);
});

test('map dimensions are recalculated when the layout changes', () => {
  act(() => root.render(<MapContainer {...baseProps} />));
  act(() => state.observers[0].callback());
  expect(state.maps[0].invalidateSize).toHaveBeenCalledWith({ pan: false });
});
