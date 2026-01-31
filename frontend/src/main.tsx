import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { useUIStore } from './stores/uiStore';
import './styles/index.css';

// Initialize dark mode before render to prevent flash
// Add no-transitions class to prevent transition flash on initial load
document.documentElement.classList.add('no-transitions');
useUIStore.getState().initDarkMode();

// Remove no-transitions class after a short delay
setTimeout(() => {
  document.documentElement.classList.remove('no-transitions');
}, 100);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
