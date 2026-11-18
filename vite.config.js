import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendTarget = 'http://127.0.0.1:8000';
const BACKEND_PREFIXES = [
  'missions', 'mission', 'image_missions', 'image_mission',
  'task4', 'upload', 'plan_mission', 'flight_params', 'drones', 'jobs',
];

function makeProxy(target) {
  return {
    target,
    changeOrigin: true,
    configure(proxy) {
      proxy.on('error', () => {
        // Backend not ready yet — suppress the default ECONNREFUSED console noise.
      });
    },
  };
}

export default defineConfig({
  root: 'dashboard',
  plugins: [react()],
  server: {
    port: 5173,
    // Every backend route prefix is listed once, here.
    proxy: Object.fromEntries(
      BACKEND_PREFIXES.map((prefix) => [`/${prefix}`, makeProxy(backendTarget)]),
    ),
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
});
