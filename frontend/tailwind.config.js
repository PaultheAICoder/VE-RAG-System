/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#003399',
          dark: '#002266',
          light: '#0055A4',
        },
        cream: '#F5F5F0',
        hero: {
          dark: '#001133',
          mid: '#002255',
        },
        corp: {
          blue: '#003399',
          'blue-light': '#0055A4',
          'blue-dark': '#002266',
          silver: '#C0C0C0',
          'silver-light': '#E8E8E8',
          'silver-dark': '#999999',
          navy: '#0d1b3a',
          'navy-light': '#1a2a4a',
          accent: '#CC6600',
        },
      },
      fontFamily: {
        heading: ['Arial', 'Helvetica', 'sans-serif'],
        body: ['Verdana', 'Tahoma', 'Arial', 'Helvetica', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'bevel': 'inset 0 1px 0 rgba(255,255,255,0.4), 1px 1px 2px rgba(0,0,0,0.15)',
        'bevel-dark': 'inset 0 1px 0 rgba(255,255,255,0.1), 1px 1px 2px rgba(0,0,0,0.3)',
        'corp': '2px 2px 4px rgba(0,0,0,0.15)',
        'corp-dark': '2px 2px 4px rgba(0,0,0,0.3)',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};
