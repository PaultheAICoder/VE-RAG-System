import React from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useAuthStore } from './stores/authStore';
import { Layout } from './components/layout';
import { ChatView } from './views/ChatView';
import { DocumentsView } from './views/DocumentsView';
import { TagsView } from './views/TagsView';
import { UsersView } from './views/UsersView';
import { SettingsView } from './views/SettingsView';
import { HealthView } from './views/HealthView';
import { RAGQualityView } from './views/RAGQualityView';

// Simple login page for development
const LoginPage = () => {
  const { login, isLoading, error, isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/chat');
    }
  }, [isAuthenticated, navigate]);

  const handleDemoLogin = async () => {
    console.log('Demo login clicked');
    try {
      console.log('Calling login...');
      await login('admin@test.com', 'npassword');
      console.log('Login successful');
    } catch (err) {
      console.error('Login error:', err);
    }
  };

  return (
    <div className="min-h-screen bg-cream dark:bg-warm-950 flex items-center justify-center px-4">
      <div className="max-w-sm w-full">
        {/* Logo and welcome */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-primary/10 dark:bg-primary/15 mb-5">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="14" cy="14" r="12" stroke="currentColor" strokeWidth="1.5" className="text-primary" />
              <path d="M9 14.5C9 14.5 11 17 14 17C17 17 19 14.5 19 14.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="text-primary" />
              <circle cx="10.5" cy="11" r="1" fill="currentColor" className="text-primary" />
              <circle cx="17.5" cy="11" r="1" fill="currentColor" className="text-primary" />
            </svg>
          </div>
          <h1 className="text-2xl font-heading font-semibold text-warm-900 dark:text-cream mb-2">
            Welcome back
          </h1>
          <p className="text-warm-500 dark:text-warm-400 text-sm">
            Sign in to AI Ready RAG
          </p>
        </div>

        {/* Card */}
        <div className="bg-white dark:bg-warm-800 p-7 rounded-2xl shadow-warm-lg dark:shadow-none dark:border dark:border-warm-700">
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 text-red-700 dark:text-red-300 p-3 rounded-xl mb-5 text-sm">
              {error}
            </div>
          )}
          <button
            onClick={handleDemoLogin}
            disabled={isLoading}
            className="w-full bg-primary hover:bg-primary-dark active:bg-primary-700 text-white font-medium py-3 px-4 rounded-xl transition-colors disabled:opacity-50 shadow-warm-md hover:shadow-warm-lg"
          >
            {isLoading ? 'Signing in...' : 'Continue with Demo Account'}
          </button>
          <p className="text-xs text-warm-400 dark:text-warm-500 mt-5 text-center">
            Backend: localhost:8506
          </p>
        </div>
      </div>
    </div>
  );
};

function App() {
  const { isAuthenticated } = useAuthStore();

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/*"
        element={
          isAuthenticated ? (
            <Layout>
              <Routes>
                <Route path="/" element={<Navigate to="/chat" replace />} />
                <Route path="/chat" element={<ChatView />} />
                <Route path="/documents" element={<DocumentsView />} />
                <Route path="/tags" element={<TagsView />} />
                <Route path="/users" element={<UsersView />} />
                <Route path="/rag-quality" element={<RAGQualityView />} />
                <Route path="/settings" element={<SettingsView />} />
                <Route path="/health" element={<HealthView />} />
              </Routes>
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
    </Routes>
  );
}

export default App;
