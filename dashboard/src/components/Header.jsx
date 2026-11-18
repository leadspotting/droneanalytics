import { useRef } from 'react';

export default function Header({
  lang,
  t,
  theme,
  mapStyle,
  onLangToggle,
  onThemeToggle,
  onMapStyleChange,
  missions,
  selectedMission,
  onMissionChange,
  onUpload,
  uploadStatus,
}) {
  const fileRef = useRef(null);

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    onUpload(file);
    event.target.value = '';
  }

  return (
    <header>
      <div className="logo">
        <span className="logo-icon">DS</span>
        <div>
          <span>{t('title')}</span>
          <span className="logo-sub">{t('subtitle')}</span>
        </div>
      </div>

      <div className="header-controls">
        <div className="mission-select-wrap">
          <label>{t('missionLbl')}</label>
          <select value={selectedMission} onChange={(event) => onMissionChange(event.target.value)}>
            <option value="">{t('selectOpt')}</option>
            {missions.map((mission) => (
              <option key={mission.mission_id} value={mission.mission_id}>
                {mission.mission_id} - {mission.drone_id} ({mission.waypoints})
              </option>
            ))}
          </select>
        </div>

        <div className="mission-select-wrap">
          <label>{lang === 'he' ? 'מפה' : 'Map'}</label>
          <select value={mapStyle} onChange={(event) => onMapStyleChange(event.target.value)}>
            <option value="light">{lang === 'he' ? 'בהיר' : 'Light'}</option>
            <option value="dark">{lang === 'he' ? 'כהה' : 'Dark'}</option>
            <option value="satellite">{lang === 'he' ? 'לוויין' : 'Satellite'}</option>
            <option value="terrain">{lang === 'he' ? 'טופוגרפי' : 'Terrain'}</option>
          </select>
        </div>

        <div className="upload-zone">
          <button type="button" className="upload-btn" onClick={() => fileRef.current?.click()}>
            {t('btnUpload')}
          </button>
          <input
            ref={fileRef}
            type="file"
            id="fileInput"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          {uploadStatus && (
            <span id="uploadStatus">{uploadStatus}</span>
          )}
        </div>

        <button type="button" className="theme-toggle" onClick={onThemeToggle}>
          <span>{theme === 'dark' ? (lang === 'he' ? 'מצב בהיר' : 'Light Mode') : (lang === 'he' ? 'מצב כהה' : 'Dark Mode')}</span>
        </button>

        <button type="button" className="lang-toggle" onClick={onLangToggle}>
          <span>{lang === 'en' ? 'EN' : 'HE'}</span>
        </button>
      </div>
    </header>
  );
}
