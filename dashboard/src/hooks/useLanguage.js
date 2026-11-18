import { useState, useCallback } from 'react';
import dict from '../i18n/dict';

/**
 * Custom hook for i18n language management.
 * Returns: lang, t(), toggleLang()
 */
export function useLanguage() {
  const [lang, setLang] = useState('en');

  const toggleLang = useCallback(() => {
    setLang(prev => {
      const next = prev === 'en' ? 'he' : 'en';
      document.documentElement.setAttribute('dir', next === 'he' ? 'rtl' : 'ltr');
      document.documentElement.setAttribute('lang', next);
      return next;
    });
  }, []);

  const t = useCallback((key) => dict[lang][key] || key, [lang]);

  return { lang, t, toggleLang };
}
