import { useState } from 'react';

const TASK_COPY = {
  en: {
    footprint: {
      title: 'Footprint Extraction',
      summary: 'Derives ground coverage polygons from telemetry and camera parameters.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Load a CSV mission, select a mission from the header, and review the map coverage plus waypoint telemetry in the sidebar.',
    },
    heatmap: {
      title: 'Coverage Heatmap',
      summary: 'Shows how often each mission cell was observed.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Open a mission with telemetry data and compare the heatmap against the footprints to identify weak or redundant coverage areas.',
    },
    mosaic: {
      title: 'Mosaic Preview',
      summary: 'Builds a geospatial image preview from captured imagery.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Load an image mission, choose Single or Grouped view, wait for the preview to render, then inspect the aligned image and export when needed.',
    },
    reference: {
      title: 'Reference Image',
      summary: 'Loads a georeferenced orthomosaic and lets you click it to focus the map.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Upload a GeoTIFF or JPG with GPS EXIF, wait for the preview to load, then click any point in the image to jump to the matching map location.',
    },
    projection: {
      title: 'Pixel to GPS',
      summary: 'Projects between image pixels and geographic coordinates.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Choose CSV Mission or Image Mission, enter pixel or GPS values, and run the conversion to verify image-to-map alignment.',
    },
    altitude: {
      title: 'Flight Calculator',
      summary: 'Calculates a recommended flight profile from camera geometry and target GSD.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Fill in the camera sensor, focal length, resolution, and target GSD values, then generate a flight profile to get altitude and speed recommendations.',
    },
    planner: {
      title: 'Mission Planner',
      summary: 'Plans a waypoint route from a drawn area of interest.',
      usageTitle: 'How to use',
      usageButton: 'How to use',
      usage: 'Draw a polygon on the map, review the generated route, optionally drag visible waypoints, then save and export the final flight plan.',
    },
  },
  he: {
    footprint: {
      title: 'חילוץ כיסוי',
      summary: 'מחשב פוליגוני כיסוי קרקעי מתוך טלמטריה ופרמטרי מצלמה.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'טען משימת CSV, בחר משימה מהכותרת העליונה, ובדוק על המפה את הכיסוי ואת נתוני הטלמטריה בסרגל הצד.',
    },
    heatmap: {
      title: 'מפת כיסוי',
      summary: 'מציגה כמה פעמים כל תא במשימה נצפה.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'פתח משימה עם נתוני טלמטריה והשווה בין מפת הכיסוי לבין ה-footprints כדי לזהות אזורים חלשים או חפיפות מיותרות.',
    },
    mosaic: {
      title: 'תצוגת מוזאיקה',
      summary: 'מרכיבה תצוגת preview גיאוגרפית מהתמונות שנקלטו.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'טען משימת תמונות, בחר תצוגת בודד או מקובץ, המתן לטעינת ה-preview, ואז בדוק את היישור וייצא במידת הצורך.',
    },
    reference: {
      title: 'תמונת רפרנס',
      summary: 'טוענת תמונה גיאוגרפית ומאפשרת קליק שממקם את המפה.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'העלה GeoTIFF או JPG עם GPS EXIF, המתן לטעינת התצוגה המקדימה, ואז לחץ על נקודה בתמונה כדי לקפוץ למיקום המתאים במפה.',
    },
    projection: {
      title: 'פיקסל ל-GPS',
      summary: 'ממיר בין פיקסלים בתמונה לבין קואורדינטות גיאוגרפיות.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'בחר עבודה מול משימת CSV או משימת תמונות, הזן ערכי פיקסל או GPS, והריץ המרה כדי לבדוק התאמה בין תמונה למפה.',
    },
    altitude: {
      title: 'חישוב טיסה',
      summary: 'מחשב פרופיל טיסה מומלץ לפי גאומטריית מצלמה ו-GSD יעד.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'מלא נתוני חיישן, אורך מוקד, רזולוציה ו-GSD יעד, ואז חשב פרופיל טיסה כדי לקבל המלצת גובה ומהירות.',
    },
    planner: {
      title: 'מתכנן משימה',
      summary: 'בונה מסלול מנקודת עניין שמצוירת על המפה.',
      usageTitle: 'איך משתמשים',
      usageButton: 'איך להשתמש',
      usage: 'צייר פוליגון על המפה, בדוק את המסלול שנבנה, גרור נקודות במידת הצורך, ואז שמור וייצא את מסלול הטיסה.',
    },
  },
};

export function TaskSectionHeader({ lang, taskKey }) {
  const [isOpen, setIsOpen] = useState(false);
  const copyByLang = TASK_COPY[lang] ?? TASK_COPY.en;
  const task = copyByLang[taskKey];

  if (!task) return null;

  return (
    <div className="task-section-header">
      <button
        type="button"
        className="task-section-toggle"
        onClick={() => setIsOpen((currentValue) => !currentValue)}
        aria-expanded={isOpen}
      >
        <span className="task-section-title">{task.title}</span>
        <span className={`task-section-cta${isOpen ? ' task-section-cta-open' : ''}`}>
          {task.usageButton}
        </span>
      </button>
      <p className="task-section-summary">{task.summary}</p>
      {isOpen && (
        <div className="task-section-help">
          <div className="task-section-help-title">{task.usageTitle}</div>
          <p className="task-section-help-body">{task.usage}</p>
        </div>
      )}
    </div>
  );
}
