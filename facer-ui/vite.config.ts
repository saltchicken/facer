import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(), // Tailwind v4 plugin
  ],
  server: {
    proxy: {
      // Proxy API requests to your Python backend during development
      '/analyze': 'http://127.0.0.1:8000',
      '/faces': 'http://127.0.0.1:8000',
    }
  }
})
