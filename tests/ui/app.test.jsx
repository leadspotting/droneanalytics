import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const api = vi.hoisted(() => Object.fromEntries(`createCustomDrone fetchDrones fetchImageMissions fetchImageMosaic fetchMissionData fetchMissions fetchTask4Reference planMission recommendFlightParams projectGpsToPixel projectPixelToGps uploadTask4Reference uploadImageMissionFiles uploadMissionCsv`.split(' ').map(name => [name, vi.fn()])));
vi.mock('../../dashboard/src/services/api', () => api);
vi.mock('../../dashboard/src/components/Header', () => ({ default: (props) => <div>
  <select id="missions" value={props.selectedMission} onChange={event => props.onMissionChange(event.target.value)}>
    <option value="" />{props.missions.map(m => <option key={m.mission_id} value={m.mission_id}>{m.mission_id}</option>)}
  </select>
  <button id="upload" onClick={() => props.onUpload(new File(['csv'], 'mission.csv'))}>Upload</button>
</div> }));
vi.mock('../../dashboard/src/components/Sidebar/Sidebar', () => ({ default: () => null }));
vi.mock('../../dashboard/src/components/MapContainer', () => ({ default: ({ fpData }) => <output id="map-data">{fpData?.mission_id}:{fpData?.count}</output> }));
vi.mock('../../dashboard/src/hooks/useLanguage', () => {
  const t = key => key;
  return { useLanguage: () => ({ lang: 'en', t, toggleLang: () => {} }) };
});
vi.mock('../../dashboard/src/hooks/useTheme', () => ({ useTheme: () => ({ theme: 'dark', toggleTheme: () => {} }) }));
import App from '../../dashboard/src/App.jsx';

let root, container;
const response = (mission_id, count) => ({ footprints: { mission_id, count }, heatmap: {}, mosaic: {} });
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  for (const method of Object.values(api)) method.mockReset();
  api.fetchMissions.mockResolvedValue([{ mission_id: 'M1' }, { mission_id: 'M2' }]);
  api.fetchDrones.mockResolvedValue({ drones: [] });
  api.fetchImageMissions.mockResolvedValue([]);
  api.fetchTask4Reference.mockResolvedValue(null);
  api.fetchMissionData.mockResolvedValue(response('M1', 1));
  api.uploadMissionCsv.mockResolvedValue({ total_rows: 2 });
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
});
afterEach(() => { act(() => root.unmount()); container.remove(); vi.useRealTimers(); });

test('uploading the same mission ID refreshes footprints and map data', async () => {
  api.fetchMissionData.mockResolvedValueOnce(response('M1', 1)).mockResolvedValueOnce(response('M1', 2));
  await act(async () => root.render(<App />));
  expect(container.querySelector('#map-data').textContent).toBe('M1:1');
  await act(async () => container.querySelector('#upload').click());
  expect(container.querySelector('#map-data').textContent).toBe('M1:2');
  expect(api.fetchMissionData).toHaveBeenCalledTimes(2);
});

test('a slow earlier mission response cannot replace the latest selection', async () => {
  let finishFirst;
  api.fetchMissionData.mockImplementationOnce(() => new Promise(resolve => { finishFirst = resolve; }))
    .mockResolvedValueOnce(response('M2', 5));
  await act(async () => root.render(<App />));
  const firstSignal = api.fetchMissionData.mock.calls[0][1].signal;
  await act(async () => {
    const select = container.querySelector('#missions');
    select.value = 'M2'; select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  expect(firstSignal.aborted).toBe(true);
  expect(container.querySelector('#map-data').textContent).toBe('M2:5');
  await act(async () => finishFirst(response('M1', 99)));
  expect(container.querySelector('#map-data').textContent).toBe('M2:5');
});

test('initial mission data loads when the backend starts after the frontend', async () => {
  vi.useFakeTimers();
  api.fetchMissions.mockRejectedValueOnce(new Error('backend starting'))
    .mockResolvedValueOnce([{ mission_id: 'M1' }]);
  await act(async () => root.render(<App />));
  expect(api.fetchMissionData).not.toHaveBeenCalled();
  await act(async () => vi.advanceTimersByTimeAsync(1000));
  expect(api.fetchMissions).toHaveBeenCalledTimes(2);
  expect(container.querySelector('#map-data').textContent).toBe('M1:1');
});
