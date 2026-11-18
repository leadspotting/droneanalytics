/**
 * Stats sub-component — 4 stat cards + coverage bar
 *
 * Props:
 *  fpData – footprints API response (or null)
 *  hmData – heatmap API response (or null)
 *  t      – translator function
 */
export default function Stats({ fpData, hmData, t }) {
  const fps = fpData?.footprints;

  const waypoints = fps ? fps.length : '—';
  const avgAlt    = fps ? (fps.reduce((a, b) => a + b.altitude_m, 0) / fps.length).toFixed(0) : '—';
  const avgBat    = fps ? (fps.reduce((a, b) => a + parseFloat(b.battery_pct), 0) / fps.length).toFixed(1) + '%' : '—';
  const avgSig    = fps ? (fps.reduce((a, b) => a + parseFloat(b.signal_pct), 0) / fps.length).toFixed(1) + '%' : '—';
  const coveragePct = hmData ? hmData.coverage_pct : null;
  const maxOverlap  = hmData ? hmData.max_overlap : '—';

  return (
    <>
      <div className="stats-grid">
        <div className="stat-card blue">
          <div className="stat-val">{waypoints}</div>
          <div className="stat-label">{t('statWp')}</div>
        </div>
        <div className="stat-card cyan">
          <div className="stat-val">{avgAlt}</div>
          <div className="stat-label">{t('statAlt')}</div>
        </div>
        <div className="stat-card green">
          <div className="stat-val">{avgBat}</div>
          <div className="stat-label">{t('statBat')}</div>
        </div>
        <div className="stat-card yellow">
          <div className="stat-val">{avgSig}</div>
          <div className="stat-label">{t('statSig')}</div>
        </div>
      </div>

      {/* Coverage bar */}
      <div className="coverage-bar-wrap" style={{ marginTop: 14 }}>
        <div className="coverage-label">
          <span style={{ color: 'var(--muted)', fontSize: 12 }}>{t('lblCoverage')}</span>
          <span style={{ fontWeight: 600, fontSize: 12 }}>
            {coveragePct !== null ? `${coveragePct}%` : '—'}
          </span>
        </div>
        <div className="bar-track">
          <div
            className="bar-fill"
            style={{ width: coveragePct !== null ? `${Math.min(coveragePct, 100)}%` : '0%' }}
          />
        </div>
      </div>
      <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>
        {t('lblMaxOverlap')} <span style={{ color: 'var(--accent2)', fontWeight: 600 }}>{maxOverlap}</span>x
      </div>
    </>
  );
}
