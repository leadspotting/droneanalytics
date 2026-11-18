import { useState } from 'react';

const INITIAL_VALUES = {
  sensorWidthMm: '13.2',
  sensorHeightMm: '8.8',
  focalLengthMm: '8.8',
  resolutionWidthPx: '5472',
  resolutionHeightPx: '3648',
  targetGsdCmPx: '2.5',
  overlapV: '0.7',
  captureIntervalS: '2.0',
};

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : '-';
}

export default function GsdCalculator({ lang, selectedDrone, onRecommendFlightParams }) {
  const [values, setValues] = useState(INITIAL_VALUES);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const copy = lang === 'he'
    ? {
        title: 'מחשבון GSD',
        subtitle: 'מפיק פרופיל טיסה שאפשר להעביר הלאה כהמלצת גובה ומהירות.',
        sensorWidth: 'רוחב חיישן (מ״מ)',
        sensorHeight: 'גובה חיישן (מ״מ)',
        focalLength: 'אורך מוקד (מ״מ)',
        resolutionWidth: 'רזולוציה רוחבית (px)',
        resolutionHeight: 'רזולוציה אנכית (px)',
        targetGsd: 'GSD יעד (ס״מ/פיקסל)',
        overlapV: 'חפיפה אנכית',
        captureInterval: 'מרווח צילום (שניות)',
        altitude: 'גובה טיסה מומלץ',
        speed: 'מהירות מומלצת',
        compute: 'חשב פרופיל טיסה',
        export: 'הורד פרופיל',
        usingDrone: 'רחפן פעיל',
        noDrone: 'ללא רחפן נבחר',
        explanation: 'הסבר',
        note: 'ההמלצה מחושבת בשרת ומתחשבת ב-GSD, חפיפה ומגבלות הרחפן כשיש רחפן נבחר.',
        invalid: 'יש למלא ערכים חיוביים בכל השדות.',
        groundSpeed: 'מהירות קרקע',
      }
    : {
        title: 'GSD Calculator',
        subtitle: 'Builds a flight profile that can be passed forward as altitude and speed guidance.',
        sensorWidth: 'Sensor width (mm)',
        sensorHeight: 'Sensor height (mm)',
        focalLength: 'Focal length (mm)',
        resolutionWidth: 'Image width (px)',
        resolutionHeight: 'Image height (px)',
        targetGsd: 'Target GSD (cm/px)',
        overlapV: 'Vertical overlap',
        captureInterval: 'Capture interval (s)',
        altitude: 'Recommended altitude',
        speed: 'Recommended speed',
        compute: 'Generate flight profile',
        export: 'Download profile',
        usingDrone: 'Active drone',
        noDrone: 'No drone selected',
        explanation: 'Explanation',
        note: 'Recommendation is computed on the server and respects GSD, overlap, and selected drone limits when available.',
        invalid: 'Fill every field with a positive value.',
        groundSpeed: 'Ground speed',
      };

  function updateValue(key, nextValue) {
    setValues((currentValues) => ({
      ...currentValues,
      [key]: nextValue,
    }));
  }

  async function handleCompute() {
    const payload = {
      target_gsd_cm_px: Number(values.targetGsdCmPx),
      sensor_width_mm: Number(values.sensorWidthMm),
      sensor_height_mm: Number(values.sensorHeightMm),
      focal_length_mm: Number(values.focalLengthMm),
      image_width_px: Number(values.resolutionWidthPx),
      image_height_px: Number(values.resolutionHeightPx),
      overlap_v: Number(values.overlapV),
      capture_interval_s: Number(values.captureIntervalS),
      drone_id: selectedDrone?.id ?? null,
    };

    const isValid = Object.entries(payload).every(([key, value]) => {
      if (key === 'drone_id') return true;
      return Number.isFinite(value) && value > 0;
    });

    if (!isValid) {
      setError(copy.invalid);
      setResult(null);
      return;
    }

    setLoading(true);
    setError('');
    try {
      const recommendation = await onRecommendFlightParams(payload);
      setResult(recommendation);
    } catch (requestError) {
      setResult(null);
      setError(requestError.message ?? copy.invalid);
    } finally {
      setLoading(false);
    }
  }

  function handleExport() {
    if (!result) return;

    const content = JSON.stringify(result, null, 2);
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'flight_profile.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="gsd-card">
      <div className="gsd-head">
        <div className="gsd-title">{copy.title}</div>
        <p className="gsd-subtitle">{copy.subtitle}</p>
      </div>

      <div className="gsd-grid">
        <label className="gsd-field">
          <span>{copy.sensorWidth}</span>
          <input value={values.sensorWidthMm} onChange={(event) => updateValue('sensorWidthMm', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.sensorHeight}</span>
          <input value={values.sensorHeightMm} onChange={(event) => updateValue('sensorHeightMm', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.focalLength}</span>
          <input value={values.focalLengthMm} onChange={(event) => updateValue('focalLengthMm', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.resolutionWidth}</span>
          <input value={values.resolutionWidthPx} onChange={(event) => updateValue('resolutionWidthPx', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.resolutionHeight}</span>
          <input value={values.resolutionHeightPx} onChange={(event) => updateValue('resolutionHeightPx', event.target.value)} />
        </label>
        <label className="gsd-field gsd-field-wide">
          <span>{copy.targetGsd}</span>
          <input value={values.targetGsdCmPx} onChange={(event) => updateValue('targetGsdCmPx', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.overlapV}</span>
          <input value={values.overlapV} onChange={(event) => updateValue('overlapV', event.target.value)} />
        </label>
        <label className="gsd-field">
          <span>{copy.captureInterval}</span>
          <input value={values.captureIntervalS} onChange={(event) => updateValue('captureIntervalS', event.target.value)} />
        </label>
      </div>

      <div className="gsd-meta">
        {copy.usingDrone}: <strong>{selectedDrone?.model ?? copy.noDrone}</strong>
      </div>

      <div className="gsd-actions">
        <button type="button" className="projection-btn" onClick={handleCompute} disabled={loading}>
          {loading ? '...' : copy.compute}
        </button>
        <button type="button" className="projection-btn projection-btn-secondary" onClick={handleExport} disabled={!result}>
          {copy.export}
        </button>
      </div>

      {error && <div className="gsd-warning">{error}</div>}

      {result && (
        <div className="gsd-results">
          <div className="gsd-result-card">
            <span>{copy.altitude}</span>
            <strong>{formatNumber(result.recommended_altitude_m)} m</strong>
          </div>
          <div className="gsd-result-card">
            <span>{copy.speed}</span>
            <strong>{formatNumber(result.recommended_speed_m_s, 2)} m/s</strong>
          </div>
          <div className="gsd-meta">
            {copy.groundSpeed}: {formatNumber(result.recommended_speed_kmh)} km/h
          </div>
          <div className="gsd-meta">
            {copy.explanation}: {result.explanation}
          </div>
        </div>
      )}

      <p className="gsd-note">{copy.note}</p>
    </div>
  );
}
