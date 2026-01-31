# AI Ready PDX — Complete Style Guide

## Brand Overview

**Company:** AI Ready PDX  
**Tagline:** Portland's AI Consulting Partner for Small Business  
**Fonts:** Poppins (headings), Inter (body)  
**Primary Color:** #2A9D8F (Teal)

---

## 1. Color Palette

### Light Mode
| Token | Hex | Usage |
|-------|-----|-------|
| Primary | `#2A9D8F` | CTAs, links, icons |
| Primary Dark | `#238578` | Hover states |
| Primary Light | `#3AB4A5` | Highlights |
| Background | `#FFFFFF` | Main background |
| Background Alt | `#F9FAFB` | Section backgrounds |
| Cream | `#FDF8F3` | Warm accent sections |
| Hero Dark | `#1E3A3A` | Hero gradient start |
| Hero Mid | `#2D4A4A` | Hero gradient end |
| Text | `#1A1A1A` | Headings |
| Text Secondary | `#374151` | Body text |
| Text Muted | `#6B7280` | Captions, labels |
| Border | `#E5E7EB` | Dividers, card borders |

### Dark Mode
| Token | Hex | Usage |
|-------|-----|-------|
| Primary | `#3AB4A5` | CTAs, links, icons |
| Primary Dark | `#2A9D8F` | Hover states |
| Primary Light | `#4ECDC4` | Highlights |
| Background | `#0F172A` | Main background |
| Background Alt | `#1E293B` | Card backgrounds |
| Background Tertiary | `#334155` | Elevated surfaces |
| Text | `#F8FAFC` | Headings |
| Text Secondary | `#CBD5E1` | Body text |
| Text Muted | `#94A3B8` | Captions, labels |
| Border | `#334155` | Dividers, card borders |

---

## 2. Typography

### Font Imports
```html
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
```

### Font Families
```css
--font-heading: 'Poppins', system-ui, sans-serif;
--font-body: 'Inter', system-ui, sans-serif;
```

### Type Scale
| Name | Size | Weight | Usage |
|------|------|--------|-------|
| Display | 3rem (48px) | 700 | Hero headlines |
| H1 | 2.25rem (36px) | 700 | Page titles |
| H2 | 1.875rem (30px) | 600 | Section headers |
| H3 | 1.5rem (24px) | 600 | Card titles |
| H4 | 1.25rem (20px) | 600 | Subsections |
| Body | 1rem (16px) | 400 | Paragraphs |
| Small | 0.875rem (14px) | 400 | Labels, captions |
| XS | 0.75rem (12px) | 500 | Badges, tags |

---

## 3. Spacing Scale

| Token | Value | Pixels |
|-------|-------|--------|
| 1 | 0.25rem | 4px |
| 2 | 0.5rem | 8px |
| 3 | 0.75rem | 12px |
| 4 | 1rem | 16px |
| 5 | 1.25rem | 20px |
| 6 | 1.5rem | 24px |
| 8 | 2rem | 32px |
| 10 | 2.5rem | 40px |
| 12 | 3rem | 48px |
| 16 | 4rem | 64px |
| 20 | 5rem | 80px |
| 24 | 6rem | 96px |

---

## 4. Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| sm | 4px | Badges, small elements |
| md | 8px | Buttons, inputs |
| lg | 12px | Cards |
| xl | 16px | Modals, large cards |
| 2xl | 24px | Hero sections |
| full | 9999px | Pills, avatars |

---

## 5. Shadows

### Light Mode
```css
--shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
--shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
--shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1);
--shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1);
```

### Dark Mode
```css
--shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
--shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
--shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5);
--shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.5);
```

---

## 6. Logo SVG

```svg
<svg width="120" height="66" viewBox="0 0 120 66" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Main arc -->
  <path d="M8 32 Q60 2 112 32" stroke="#2A9D8F" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- Left tower -->
  <path d="M20 58 L20 24 L26 16 L32 24 L32 58" stroke="#2A9D8F" stroke-width="1.5" fill="none"/>
  <line x1="26" y1="16" x2="26" y2="58" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="20" y1="28" x2="32" y2="28" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="20" y1="36" x2="32" y2="36" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="20" y1="44" x2="32" y2="44" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="20" y1="52" x2="32" y2="52" stroke="#2A9D8F" stroke-width="1.5"/>
  <rect x="22" y="10" width="8" height="6" stroke="#2A9D8F" stroke-width="1.5" fill="none"/>
  <line x1="26" y1="6" x2="26" y2="10" stroke="#2A9D8F" stroke-width="1"/>
  <!-- Right tower -->
  <path d="M88 58 L88 24 L94 16 L100 24 L100 58" stroke="#2A9D8F" stroke-width="1.5" fill="none"/>
  <line x1="94" y1="16" x2="94" y2="58" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="88" y1="28" x2="100" y2="28" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="88" y1="36" x2="100" y2="36" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="88" y1="44" x2="100" y2="44" stroke="#2A9D8F" stroke-width="1.5"/>
  <line x1="88" y1="52" x2="100" y2="52" stroke="#2A9D8F" stroke-width="1.5"/>
  <rect x="90" y="10" width="8" height="6" stroke="#2A9D8F" stroke-width="1.5" fill="none"/>
  <line x1="94" y1="6" x2="94" y2="10" stroke="#2A9D8F" stroke-width="1"/>
  <!-- Vertical cables -->
  <line x1="40" y1="12" x2="40" y2="46" stroke="#2A9D8F" stroke-width="1.2"/>
  <line x1="50" y1="7" x2="50" y2="46" stroke="#2A9D8F" stroke-width="1.2"/>
  <line x1="60" y1="4" x2="60" y2="46" stroke="#2A9D8F" stroke-width="1.2"/>
  <line x1="70" y1="7" x2="70" y2="46" stroke="#2A9D8F" stroke-width="1.2"/>
  <line x1="80" y1="12" x2="80" y2="46" stroke="#2A9D8F" stroke-width="1.2"/>
  <!-- Center span -->
  <rect x="36" y="46" width="48" height="6" stroke="#2A9D8F" stroke-width="1.5" fill="none" rx="1"/>
  <!-- Deck -->
  <line x1="4" y1="58" x2="116" y2="58" stroke="#2A9D8F" stroke-width="2"/>
</svg>
```

---

## 7. CSS Variables

```css
/* AI Ready PDX - CSS Custom Properties */
:root {
  /* Primary Colors */
  --color-primary: #2A9D8F;
  --color-primary-dark: #238578;
  --color-primary-light: #3AB4A5;
  
  /* Background Colors */
  --color-bg: #FFFFFF;
  --color-bg-secondary: #F9FAFB;
  --color-bg-tertiary: #F3F4F6;
  --color-bg-cream: #FDF8F3;
  --color-hero-start: #1E3A3A;
  --color-hero-end: #2D4A4A;
  
  /* Text Colors */
  --color-text: #1A1A1A;
  --color-text-secondary: #374151;
  --color-text-muted: #6B7280;
  
  /* Border Colors */
  --color-border: #E5E7EB;
  --color-border-dark: #D1D5DB;
  
  /* Typography */
  --font-heading: 'Poppins', system-ui, sans-serif;
  --font-body: 'Inter', system-ui, sans-serif;
  
  /* Spacing */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;
  --space-16: 4rem;
  --space-24: 6rem;
  
  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --radius-2xl: 24px;
  --radius-full: 9999px;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1);
}

/* Dark Mode */
[data-theme="dark"],
.dark {
  --color-primary: #3AB4A5;
  --color-primary-dark: #2A9D8F;
  --color-primary-light: #4ECDC4;
  
  --color-bg: #0F172A;
  --color-bg-secondary: #1E293B;
  --color-bg-tertiary: #334155;
  
  --color-text: #F8FAFC;
  --color-text-secondary: #CBD5E1;
  --color-text-muted: #94A3B8;
  
  --color-border: #334155;
  --color-border-dark: #475569;
  
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5);
}
```

---

## 8. Tailwind Config

```js
// tailwind.config.js
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2A9D8F',
          dark: '#238578',
          light: '#3AB4A5',
        },
        cream: '#FDF8F3',
        hero: {
          dark: '#1e3a3a',
          mid: '#2d4a4a',
        },
      },
      fontFamily: {
        heading: ['Poppins', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
```
