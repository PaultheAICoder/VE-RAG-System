/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#E8A0BF',
          dark: '#D4789E',
          light: '#F0B8D0',
        },
        cream: '#FFFAF5',
        blush: '#FFF5F7',
        rose: {
          50: '#FFF5F7',
          100: '#FFE8EF',
          200: '#F5C6D0',
          300: '#F0B8D0',
          400: '#E8A0BF',
          500: '#D4789E',
          600: '#C0608A',
          700: '#A34D73',
          800: '#86395C',
          900: '#6A2848',
        },
        plum: {
          800: '#2D1B2E',
          850: '#261726',
          900: '#1F121F',
          950: '#180E18',
        },
        mauve: {
          700: '#4A2D4A',
          800: '#3A2235',
          900: '#2D1B2E',
        },
        hero: {
          dark: '#3A2235',
          mid: '#4A2D4A',
        },
      },
      fontFamily: {
        heading: ['Quicksand', 'system-ui', 'sans-serif'],
        body: ['Nunito', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'warm': '0 2px 12px 0 rgba(232, 160, 191, 0.12)',
        'warm-md': '0 4px 16px 0 rgba(232, 160, 191, 0.18)',
        'warm-lg': '0 8px 24px 0 rgba(232, 160, 191, 0.25)',
        'warm-glow': '0 4px 14px 0 rgba(232, 160, 191, 0.4)',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};
