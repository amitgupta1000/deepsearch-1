import React from 'react';
import {
  DocumentTextIcon,
  ClockIcon,
  ExclamationCircleIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';
import { ArrowPathIcon } from '@heroicons/react/24/solid';
import { useResearch } from '../context/ResearchContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ResultsDisplay: React.FC = () => {
  const { state, clearResults } = useResearch();
  const { isLoading, result, error } = state;

  const downloadReport = async (contentType: 'analysis' | 'appendix') => {
    if (!result) {
      alert('Result not available, cannot download file.');
      return;
    }

    // Use the exact filename provided by the backend
    const filename = contentType === 'analysis' ? result.analysis_filename : result.appendix_filename;

    if (!filename) {
      alert(`The ${contentType} report is not available for download.`);
      return;
    }

    try {
      // Use the generic download endpoint with the correct filename
      const response = await fetch(`/api/research/${result.session_id}/download?content_type=${contentType}`);
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename; // The browser will use this name for the downloaded file
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      console.log(`Successfully downloaded ${contentType} file: ${filename}`);
    } catch (err) {
      console.error(`Error downloading ${contentType} file:`, err);
      alert(`Failed to download file. Please try again.`);
    }
  };

  if (isLoading) {
    return (
      <div className="card p-8 shadow-lg">
        <div className="flex flex-col items-center space-y-4">
          <ArrowPathIcon className="w-12 h-12 text-primary-600 animate-spin" />
          <h3 className="text-xl font-semibold text-gray-900">Research in Progress</h3>
          <p className="text-gray-600">Our AI is analyzing sources and generating your report...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 border-red-200 bg-red-50 shadow-lg">
        <div className="flex items-start space-x-4">
          <ExclamationCircleIcon className="w-6 h-6 text-red-600 mt-1" />
          <div>
            <h3 className="text-xl font-semibold text-red-900">Research Error</h3>
            <p className="text-red-700 mt-2">{error}</p>
            <button onClick={clearResults} className="btn-primary bg-red-600 hover:bg-red-700 mt-4">
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  return (
    <div className="space-y-8">
      {/* Success Header */}
      <div className="card p-6 bg-gradient-to-r from-green-50 to-blue-50 border-green-200 shadow-lg">
        <div className="flex items-center space-x-4">
          <CheckCircleIcon className="w-8 h-8 text-green-600" />
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Research Complete!</h2>
            <p className="text-gray-600">Your comprehensive report is ready below.</p>
          </div>
        </div>
      </div>

      {/* Actions & New Research */}
      <div className="card p-4 shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
                <ClockIcon className="w-5 h-5" />
                <span>{new Date(result.created_at).toLocaleString()}</span>
            </div>
            <div className="flex flex-wrap gap-2">
                <button onClick={() => downloadReport('analysis')} className="btn-secondary flex items-center space-x-2">
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    <span>Download Analysis</span>
                </button>
                <button onClick={() => downloadReport('appendix')} className="btn-secondary flex items-center space-x-2">
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    <span>Download Appendix</span>
                </button>
                <button onClick={clearResults} className="btn-primary">
                  New Research
                </button>
            </div>
        </div>
      </div>

      {/* Analysis Content */}
      {result.analysis_content && (
        <div className="card p-8 shadow-lg">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center space-x-3">
            <DocumentTextIcon className="w-6 h-6 text-primary-600" />
            <span>Analysis Report</span>
          </h3>
          <div className="prose prose-lg max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p>{String(children).replace(/\n\n/g, '<br /><br />')}</p>,
                ul: ({ children }) => <ul className="list-disc pl-6">{children}</ul>,
                li: ({ children }) => <li className="mb-2">{children}</li>
              }}
            >
              {result.analysis_content.replace(/\n\n/g, '\n')}
            </ReactMarkdown>
          </div>
        </div>
      )}

      {/* Appendix Content */}
      {result.appendix_content && (
        <div className="card p-8 shadow-lg">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4 flex items-center space-x-3">
            <DocumentTextIcon className="w-6 h-6 text-blue-600" />
            <span>Appendix</span>
          </h3>
          <div className="prose max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p>{String(children).replace(/\n\n/g, '<br /><br />')}</p>,
                ul: ({ children }) => <ul className="list-disc pl-6">{children}</ul>,
                li: ({ children }) => <li className="mb-2">{children}</li>
              }}
            >
              {result.appendix_content.replace(/\n\n/g, '\n')}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
