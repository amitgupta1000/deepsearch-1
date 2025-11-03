import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { AuthState, AuthContextType } from '../types';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    apiKey: null,
    userInfo: null,
    error: null
  });

  // API endpoint configuration
  const getApiUrl = () => {
    return import.meta.env.VITE_API_URL || 'http://localhost:8000';
  };

  // Set and validate API key
  const setApiKey = async (apiKey: string): Promise<boolean> => {
    try {
      setAuthState(prev => ({ ...prev, error: null }));
      
      // Validate the API key by calling the auth info endpoint
      const response = await fetch(`${getApiUrl()}/api/auth/info`, {
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey.trim()
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Invalid API key' }));
        setAuthState(prev => ({ 
          ...prev, 
          error: errorData.detail || 'Invalid API key',
          isAuthenticated: false,
          apiKey: null,
          userInfo: null
        }));
        return false;
      }

      const authData = await response.json();
      
      // Store API key in localStorage and update state
      localStorage.setItem('intellisearch_api_key', apiKey.trim());
      
      setAuthState({
        isAuthenticated: true,
        apiKey: apiKey.trim(),
        userInfo: authData.user,
        error: null
      });

      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
      setAuthState(prev => ({ 
        ...prev, 
        error: errorMessage,
        isAuthenticated: false,
        apiKey: null,
        userInfo: null
      }));
      return false;
    }
  };

  // Clear authentication
  const clearAuth = () => {
    localStorage.removeItem('intellisearch_api_key');
    setAuthState({
      isAuthenticated: false,
      apiKey: null,
      userInfo: null,
      error: null
    });
  };

  // Check if stored API key is still valid
  const checkAuth = async (): Promise<boolean> => {
    const storedApiKey = localStorage.getItem('intellisearch_api_key');
    
    if (!storedApiKey) {
      return false;
    }

    try {
      const response = await fetch(`${getApiUrl()}/api/auth/info`, {
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': storedApiKey
        }
      });

      if (response.ok) {
        const authData = await response.json();
        setAuthState({
          isAuthenticated: true,
          apiKey: storedApiKey,
          userInfo: authData.user,
          error: null
        });
        return true;
      } else {
        // Stored API key is invalid
        clearAuth();
        return false;
      }
    } catch (error) {
      clearAuth();
      return false;
    }
  };

  // Check authentication on component mount
  useEffect(() => {
    checkAuth();
  }, []);

  const contextValue: AuthContextType = {
    authState,
    setApiKey,
    clearAuth,
    checkAuth
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Custom hook for making authenticated requests
export const useAuthenticatedFetch = () => {
  const { authState, clearAuth } = useAuth();

  const authenticatedFetch = async (endpoint: string, options: RequestInit = {}) => {
    const apiKey = authState.apiKey || localStorage.getItem('intellisearch_api_key');
    
    if (!apiKey) {
      throw new Error('No API key available. Please authenticate first.');
    }

    const headers = {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey,
      ...options.headers
    };

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const response = await fetch(`${apiUrl}${endpoint}`, {
      ...options,
      headers
    });

    if (response.status === 401) {
      // API key is invalid, clear auth state
      clearAuth();
      throw new Error('Session expired. Please authenticate again.');
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    return response;
  };

  return authenticatedFetch;
};