import React from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useAuthStore } from './stores/authStore';
import { Layout } from './components/layout';
import { ChatView } from './views/ChatView';
import { DocumentsView } from './views/DocumentsView';

// Placeholder views - will be implemented in separate issues
const PlaceholderView = ({ name }: { name: string }) => (
  <div className="p-8">
    <h1 className="text-2xl font-heading font-bold">{name} View</h1>
    <p className="text-gray-600 dark:text-gray-400 mt-2">Coming soon...</p>
  </div>
);

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
      await login('admin@test.com', 'npassword2002!');
      console.log('Login successful');
    } catch (err) {
      console.error('Login error:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
      <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg max-w-md w-full">
        <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white mb-6">
          AI Ready RAG
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Login to access the system
        </p>
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 p-3 rounded-lg mb-4">
            {error}
          </div>
        )}
        <button
          onClick={handleDemoLogin}
          disabled={isLoading}
          className="w-full bg-primary hover:bg-primary-dark text-white font-semibold py-2.5 px-4 rounded-lg transition-colors disabled:opacity-50"
        >
          {isLoading ? 'Logging in...' : 'Demo Login (Admin)'}
        </button>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-4 text-center">
          Backend: localhost:8502
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
                <Route path="/tags" element={<PlaceholderView name="Tags" />} />
                <Route path="/users" element={<PlaceholderView name="Users" />} />
                <Route path="/settings" element={<PlaceholderView name="Settings" />} />
                <Route path="/health" element={<PlaceholderView name="Health" />} />
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
