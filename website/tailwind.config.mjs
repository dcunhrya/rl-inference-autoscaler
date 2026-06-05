/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        cream: {
          DEFAULT: '#FAF7F2',
          card: '#FFFDF9',
          muted: '#F0EBE3',
        },
        ink: {
          DEFAULT: '#2E2D29',
          muted: '#5F5C57',
          light: '#8A8680',
        },
        accent: {
          DEFAULT: '#8C1515',
          dim: '#6B0F0F',
          light: '#B83A3A',
          glow: 'rgba(140, 21, 21, 0.12)',
        },
        border: {
          DEFAULT: '#E8E2D8',
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
