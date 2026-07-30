import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // If your backend serves the dist folder at root (e.g. http://localhost:8000/),
  // keep base as '/'. If it serves under a subpath like '/frontend/', change this.
  base: '/',
  build: {
    // FIX: Vite 8 switched the default CSS minifier to lightningcss,
    // which corrupts backdrop-filter / -webkit-backdrop-filter.
    // Use false to disable CSS minification entirely (no esbuild needed).
    cssMinify: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/queue':    { target: 'http://localhost:8000', changeOrigin: true },
      '/health':   { target: 'http://localhost:8000', changeOrigin: true },
      '/export':   { target: 'http://localhost:8000', changeOrigin: true },
      '/account':  { target: 'http://localhost:8000', changeOrigin: true },
      '/auth':     { target: 'http://localhost:8000', changeOrigin: true },
      '/dashboard':{ target: 'http://localhost:8000', changeOrigin: true },
      '/records':  { target: 'http://localhost:8000', changeOrigin: true },
      '/vitals':   { target: 'http://localhost:8000', changeOrigin: true },
      '/metrics':  { target: 'http://localhost:8000', changeOrigin: true },
      '/readyz':   { target: 'http://localhost:8000', changeOrigin: true },
      '/tts':      { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
