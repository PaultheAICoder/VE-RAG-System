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
    <div className="min-h-screen bg-gradient-to-br from-blush via-cream to-rose-100 dark:from-plum-950 dark:via-plum-900 dark:to-mauve-900 flex items-center justify-center p-4">
      {/* Decorative background circles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-rose-200/30 dark:bg-rose-800/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-32 -right-32 w-[500px] h-[500px] bg-primary/10 dark:bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-rose-100/40 dark:bg-rose-900/10 rounded-full blur-2xl" />
      </div>

      <div className="relative bg-white/80 dark:bg-plum-800/80 backdrop-blur-sm p-10 rounded-3xl shadow-warm-lg max-w-md w-full border border-rose-100 dark:border-mauve-700">
        {/* Welcome icon */}
        <div className="flex justify-center mb-5">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-rose-500 flex items-center justify-center shadow-warm-glow">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
            </svg>
          </div>
        </div>

        <h1 className="text-2xl font-heading font-bold text-rose-800 dark:text-rose-200 mb-2 text-center">
          Welcome Back
        </h1>
        <p className="text-rose-600/70 dark:text-rose-300/60 mb-8 text-center text-sm">
          Sign in to your cozy workspace
        </p>
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 p-3 rounded-xl mb-4 text-sm">
            {error}
          </div>
        )}
        <button
          onClick={handleDemoLogin}
          disabled={isLoading}
          className="w-full bg-gradient-to-r from-primary to-rose-500 hover:from-primary-dark hover:to-rose-600 text-white font-semibold py-3 px-4 rounded-2xl transition-all duration-300 disabled:opacity-50 shadow-warm-glow hover:shadow-warm-lg hover:scale-[1.02] active:scale-[0.98]"
        >
          {isLoading ? 'Logging in...' : 'Demo Login (Admin)'}
        </button>
        <p className="text-sm text-rose-400/60 dark:text-rose-300/40 mt-6 text-center">
          Backend: localhost:8503
        </p>
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
