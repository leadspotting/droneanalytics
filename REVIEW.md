# Integration verification ? 2026-11-08

The integrated version combines the tested image engine from GitHub `main`
(`b9747c3`) with the local fixes. The experimental `1-improve-task-3` branch was
compared in an isolated copy: it had 80 failing tests, whereas GitHub `main` had
352 passing tests and no failures. The experimental branch is retained separately.

The original local snapshot is preserved as
`backup/local-before-integration-20261108`. The integration retains both commit
histories, without overwriting the existing GitHub history.

## Changes

- Restored `ImageStitcher` and its four supporting modules from the tested main
  branch. Exceptions are no longer converted into a fake black mosaic. Fixed
  numeric precision when comparing uniform images.
- Bundled Leaflet CSS with the frontend rather than loading it from a CDN.
- Verified light/dark CARTO and Esri satellite/topographic tile endpoints. Replaced
  the terrain provider whose TLS certificate failed validation. Attribution is
  displayed. Map styles and drone selection no longer recreate the map instance;
  resizing recalculates the map size, and polygon edits update mission planning.
- Retried startup requests while the Python backend is starting. Re-uploading a
  mission refreshes map data even when its ID is unchanged. Older requests are
  cancelled so they cannot replace the current selection.
- Preserved valid previous data when CSV/SRT, image or reference replacements
  fail. Image-directory installation rolls back on rename failure.
- Kept CSV/GPS validation, popup HTML escaping, mission-ID URL encoding,
  southern-hemisphere UTM correction, missing Python dependencies and the
  corrected launch/build scripts.
- Retained patched npm dependencies and added automated map/refresh component
  regressions. Tests are restricted to project test directories.
- Excluded generated packages, optional datasets, local agent settings and
  planning documents. Source, tests, lockfiles and Windows build scripts remain
  versioned. Generated binaries belong in GitHub Release assets.

## Verification

- Full Python suite: **378 passed, 18 skipped**, plus 2 passing subtests.
- `npm test`: **8 passed** (2 utility tests and 6 component tests).
- `npm run build`: passed.
- Fresh `npm ci`: passed; npm dependency audit reports no known vulnerabilities.
- End-to-end synthetic imagery: real geotagged JPEG upload, metadata extraction,
  actual stitching, PNG preview, ZIP export and GPS projection pass without
  mocking the stitcher.
- Invalid image/reference replacement and simulated install failure preserve the
  previous data.
- Production HTML/assets, sample telemetry routes and the Vite development proxy
  were checked through HTTP.

## Validation limits

The 18 skipped tests require external SEKI imagery or Task 4 reference fixtures
that are absent from this checkout. Passing synthetic and unit tests does not
establish photogrammetric accuracy on those datasets.

Visual browser verification could not be completed: the Computer Use runtime
stopped because it could not confidently identify the active browser URL. Map
component behavior, bundled assets and provider responses were tested separately.
A fresh Windows executable was not packaged/tested during this integration; the
existing local `release/` directory is an older generated output.

Basemaps require internet access. The application is configured for loopback use,
keeps mission registrations in memory, and has no authentication. Very large
imagery, polygons or extremely fine heatmap grids can require substantial memory
and computation.

Reproduce checks from the project root:

```powershell
python -m pip install -r requirements.txt
npm ci
python -m pytest -q
npm test
npm run build
```
