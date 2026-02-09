/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#D97757',
          dark: '#C4663F',
          light: '#E8956E',
          50: '#FEF5F1',
          100: '#FCEADE',
          200: '#F5C5AB',
          300: '#F0AD8D',
          400: '#E8956E',
          500: '#D97757',
          600: '#C4663F',
          700: '#A3502E',
          800: '#7D3D23',
          900: '#5C2D19',
        },
        accent: {
          DEFAULT: '#C8A86E',
          light: '#DFC99B',
          dark: '#A88B4A',
        },
        cream: {
          DEFAULT: '#FAF9F7',
          dark: '#F5F2ED',
          100: '#F0EDE8',
        },
        warm: {
          50: '#FAF9F7',
          100: '#F5F2ED',
          200: '#E8E2DA',
          300: '#D4CBC0',
          400: '#A8A29E',
          500: '#78716C',
          600: '#57534E',
          700: '#44403C',
          800: '#292524',
          850: '#231F1B',
          900: '#1C1917',
          950: '#1A1613',
        },
      },
      fontFamily: {
        heading: ['"Source Serif 4"', '"Styrene A"', 'Georgia', 'serif'],
        body: ['"DM Sans"', '"Söhne"', '"Helvetica Neue"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'warm': '0 1px 3px 0 rgba(45, 41, 38, 0.06), 0 1px 2px -1px rgba(45, 41, 38, 0.06)',
        'warm-md': '0 4px 6px -1px rgba(45, 41, 38, 0.07), 0 2px 4px -2px rgba(45, 41, 38, 0.05)',
        'warm-lg': '0 10px 15px -3px rgba(45, 41, 38, 0.08), 0 4px 6px -4px rgba(45, 41, 38, 0.04)',
        'warm-xl': '0 20px 25px -5px rgba(45, 41, 38, 0.08), 0 8px 10px -6px rgba(45, 41, 38, 0.04)',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};
