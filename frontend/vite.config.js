import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/queue':    { target: 'http://localhost:8000', changeOrigin: true },
      '/health':   { target: 'http://localhost:8000', changeOrigin: true },
      '/export':   { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/account':  { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/auth':     { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/dashboard':{ target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/records':  { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/vitals':   { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/metrics':  { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
      '/readyz':   { target: 'http://localhost:8000', changeOrigin: true },  // ← ADD
    },
  },
})