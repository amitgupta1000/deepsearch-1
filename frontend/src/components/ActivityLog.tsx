import React from 'react';
import { useResearch } from '../context/ResearchContext';
import { 
  InformationCircleIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  XCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

const ActivityLog: React.FC = () => {
  const { state } = useResearch();

  if (!state.isLoading && state.logs.length === 0) {
    return null;
  }

  const getIconForLevel = (level: string) => {
    switch (level) {
      case 'success':
        return <CheckCircleIcon className="w-4 h-4 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-4 h-4 text-yellow-500" />;
      case 'error':
        return <XCircleIcon className="w-4 h-4 text-red-500" />;
      default:
        return <InformationCircleIcon className="w-4 h-4 text-blue-500" />;
    }
  };

  const getColorForLevel = (level: string) => {
    switch (level) {
      case 'success':
        return 'text-green-800 bg-green-50 border-green-200';
      case 'warning':
        return 'text-yellow-800 bg-yellow-50 border-yellow-200';
      case 'error':
        return 'text-red-800 bg-red-50 border-red-200';
      default:
        return 'text-blue-800 bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="mt-6 bg-white rounded-lg shadow-md border border-gray-200">
      <div className="px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
          <ClockIcon className="w-5 h-5" />
          <span>Research Activity Log</span>
        </h3>
        {state.isLoading && (
          <div className="mt-2">
            <div className="flex items-center space-x-3">
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${state.progress}%` }}
                ></div>
              </div>
              <span className="text-sm text-gray-600">{state.progress}%</span>
            </div>
            <p className="text-sm text-gray-600 mt-1">{state.currentStep}</p>
          </div>
        )}
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {state.logs.length === 0 ? (
          <div className="px-6 py-4 text-gray-500 text-center">
            No activity logs yet. Start a research to see real-time progress.
          </div>
        ) : (
          <div className="space-y-2 p-4">
            {state.logs.map((log) => (
              <div
                key={log.id}
                className={`p-3 rounded-lg border text-sm ${getColorForLevel(log.level)}`}
              >
                <div className="flex items-start space-x-2">
                  {getIconForLevel(log.level)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{log.message}</span>
                      <span className="text-xs opacity-75">{log.timestamp}</span>
                    </div>
                    {log.details && (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-xs opacity-75 hover:opacity-100">
                          Show details
                        </summary>
                        <pre className="mt-1 text-xs bg-black/5 p-2 rounded overflow-auto">
                          {typeof log.details === 'string' 
                            ? log.details 
                            : JSON.stringify(log.details, null, 2)
                          }
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ActivityLog;