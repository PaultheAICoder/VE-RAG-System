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
    <div className="min-h-screen bg-white dark:bg-[#343541] flex items-center justify-center">
      <div className="max-w-sm w-full px-6">
        {/* Logo / Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-[#10A37F] mb-4">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <h1 className="text-2xl font-semibold text-[#2D2D2D] dark:text-[#ECECF1] mb-2">
            AI Ready RAG
          </h1>
          <p className="text-sm text-[#6E6E80] dark:text-[#ACACBE]">
            Login to access the system
          </p>
        </div>

        {/* Error message */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 p-3 rounded-lg mb-4 text-sm">
            {error}
          </div>
        )}

        {/* Login button */}
        <button
          onClick={handleDemoLogin}
          disabled={isLoading}
          className="w-full bg-[#10A37F] hover:bg-[#0D8A6A] text-white font-medium py-3 px-4 rounded-md transition-colors disabled:opacity-50 text-sm"
        >
          {isLoading ? 'Logging in...' : 'Demo Login (Admin)'}
        </button>

        <p className="text-xs text-[#6E6E80] dark:text-[#ACACBE] mt-6 text-center">
          Backend: localhost:8507
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
