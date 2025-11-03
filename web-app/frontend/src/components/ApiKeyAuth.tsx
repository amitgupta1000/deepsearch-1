import React, { useState } from 'react';
import { KeyIcon, ExclamationTriangleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../context/AuthContext';

export const ApiKeyAuth: React.FC = () => {
  const { authState, setApiKey, clearAuth } = useAuth();
  const [inputKey, setInputKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputKey.trim()) {
      return;
    }

    setIsLoading(true);
    const success = await setApiKey(inputKey.trim());
    setIsLoading(false);

    if (success) {
      setInputKey(''); // Clear input on success
    }
  };

  const handleSignOut = () => {
    clearAuth();
    setInputKey('');
  };

  // Show authenticated state
  if (authState.isAuthenticated && authState.userInfo) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <CheckCircleIcon className="w-5 h-5 text-green-600" />
            <div>
              <p className="text-sm font-medium text-green-800">
                Authenticated as {authState.userInfo.user_id}
              </p>
              <p className="text-xs text-green-600">
                Rate limit: {authState.userInfo.rate_limits.requests_per_hour} requests/hour
              </p>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className="text-sm text-green-700 hover:text-green-900 font-medium"
          >
            Sign Out
          </button>
        </div>
      </div>
    );
  }

  // Show authentication form
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <KeyIcon className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-900">API Key Required</h3>
      </div>
      
      <p className="text-sm text-gray-600 mb-4">
        Please enter your INTELLISEARCH API key to access the research platform.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-2">
            API Key
          </label>
          <input
            type="password"
            id="apiKey"
            value={inputKey}
            onChange={(e) => setInputKey(e.target.value)}
            placeholder="Enter your API key..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isLoading}
            required
          />
        </div>

        {authState.error && (
          <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-md">
            <ExclamationTriangleIcon className="w-5 h-5 text-red-600 flex-shrink-0" />
            <p className="text-sm text-red-700">{authState.error}</p>
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || !inputKey.trim()}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Authenticating...' : 'Authenticate'}
        </button>
      </form>

      <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
        <h4 className="text-sm font-medium text-blue-900 mb-2">Demo API Keys (for testing):</h4>
        <div className="space-y-1 text-xs text-blue-800 font-mono">
          <div 
            className="cursor-pointer hover:bg-blue-100 p-1 rounded"
            onClick={() => setInputKey('demo-key-research-123')}
          >
            demo-key-research-123 (Regular User)
          </div>
          <div 
            className="cursor-pointer hover:bg-blue-100 p-1 rounded"
            onClick={() => setInputKey('demo-key-admin-456')}
          >
            demo-key-admin-456 (Admin User)
          </div>
        </div>
        <p className="text-xs text-blue-600 mt-2">
          Click any demo key to auto-fill the input field.
        </p>
      </div>
    </div>
  );
};