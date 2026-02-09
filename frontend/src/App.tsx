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
    <div className="min-h-screen bg-white dark:bg-[#0A0A0B] flex items-center justify-center relative overflow-hidden">
      {/* Subtle gradient orb background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary-100/40 dark:bg-primary-500/[0.07] rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-primary-200/30 dark:bg-primary-400/[0.05] rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 w-full max-w-sm mx-4">
        {/* Logo and branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-primary/10 dark:bg-primary/20 mb-4">
            <div className="w-6 h-6 rounded-md bg-primary" />
          </div>
          <h1 className="text-xl font-semibold text-gray-900 dark:text-white tracking-tight">
            AI Ready RAG
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
            Intelligent document retrieval
          </p>
        </div>

        {/* Login card */}
        <div className="bg-white dark:bg-[#111113] border border-gray-200/80 dark:border-[#1E1E22] rounded-2xl p-8 shadow-elevated dark:shadow-none">
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
            Sign in to access the system
          </p>
          {error && (
            <div className="bg-red-50 dark:bg-red-900/10 border border-red-200/60 dark:border-red-800/30 text-red-600 dark:text-red-400 text-sm p-3 rounded-xl mb-4">
              {error}
            </div>
          )}
          <button
            onClick={handleDemoLogin}
            disabled={isLoading}
            className="w-full bg-primary hover:bg-primary-dark text-white font-medium py-2.5 px-4 rounded-xl transition-all duration-200 disabled:opacity-50 hover:shadow-glow active:scale-[0.98]"
          >
            {isLoading ? 'Signing in...' : 'Continue with Demo'}
          </button>
        </div>

        <p className="text-xs text-gray-400 dark:text-gray-600 mt-6 text-center">
          Backend: localhost:8504
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
