/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#10A37F',
          dark: '#0D8A6A',
          light: '#19C37D',
        },
        surface: {
          DEFAULT: '#FFFFFF',
          secondary: '#F7F7F8',
          dark: '#343541',
          'dark-secondary': '#202123',
        },
        border: {
          light: '#E5E5E5',
          dark: '#4E4F60',
        },
        chatgpt: {
          green: '#10A37F',
          'green-light': '#19C37D',
          'green-dark': '#0D8A6A',
          'dark-bg': '#343541',
          'dark-sidebar': '#202123',
          'dark-hover': '#2A2B32',
          'light-bg': '#FFFFFF',
          'light-sidebar': '#F7F7F8',
          'light-hover': '#ECECF1',
          'text-primary': '#2D2D2D',
          'text-secondary': '#6E6E80',
          'text-dark-primary': '#ECECF1',
          'text-dark-secondary': '#ACACBE',
        },
      },
      fontFamily: {
        heading: ['Söhne', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
        body: ['Söhne', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};
