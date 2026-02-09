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
    <div className="min-h-screen corp-page-bg flex items-center justify-center">
      <div className="corp-panel max-w-md w-full">
        {/* Classic gradient title bar */}
        <div className="corp-panel-header">
          <h1 className="text-lg font-bold tracking-wide font-heading">
            AI Ready RAG - System Login
          </h1>
        </div>
        <div className="p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-1 text-sm font-body">
            Welcome to the AI Ready RAG Management Portal.
          </p>
          <p className="text-gray-600 dark:text-gray-400 mb-5 text-sm font-body">
            Please authenticate to access the system.
          </p>
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 p-3 mb-4 text-sm">
              {error}
            </div>
          )}
          <button
            onClick={handleDemoLogin}
            disabled={isLoading}
            className="w-full corp-primary-btn py-2.5 px-4 text-sm disabled:opacity-50 cursor-pointer"
          >
            {isLoading ? 'Authenticating...' : 'Demo Login (Admin)'}
          </button>
          <hr className="corp-hr my-4" />
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center font-body">
            Backend Server: localhost:8505 | &copy; 2002 AI Ready RAG Corp.
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
