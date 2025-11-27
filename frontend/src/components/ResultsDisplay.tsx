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
  const [isDownloading, setIsDownloading] = React.useState<'analysis' | 'appendix' | null>(null);
  const { state, clearResults } = useResearch();
  const { isLoading, result, error } = state;

  const downloadReport = async (contentType: 'analysis' | 'appendix') => {
    setIsDownloading(contentType);

    if (!result) {
      alert('Result not available, cannot download file.');
      setIsDownloading(null);
      return;
    }

    const content = contentType === 'analysis' ? result.analysis_content : result.appendix_content;
    // Use correct property names for filenames, e.g. analysis_file_name and appendix_file_name
    const filename = contentType === 'analysis' ? result.analysis_filename : result.appendix_filename;

    if (!content || !filename) {
      alert(`The ${contentType} report content or filename is not available for download.`);
      setIsDownloading(null);
      return;
    }

    try {
      // Create a Blob from the content that's already in the state
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      console.log(`Successfully downloaded ${contentType} file: ${filename}`);
    } catch (err) {
      console.error(`Error creating download for ${contentType} file:`, err);
      alert(`Failed to create download. Please try again.`);
    } finally {
      setIsDownloading(null);
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
                <button
                  onClick={() => downloadReport('analysis')}
                  className="btn-secondary flex items-center space-x-2"
                  disabled={!!isDownloading}
                >
                  {isDownloading === 'analysis' ? (
                    <ArrowPathIcon className="w-4 h-4 animate-spin" />
                  ) : (
                    <ArrowDownTrayIcon className="w-4 h-4" />
                  )}
                  <span>{isDownloading === 'analysis' ? 'Downloading...' : 'Download Analysis'}</span>
                </button>
                <button
                  onClick={() => downloadReport('appendix')}
                  className="btn-secondary flex items-center space-x-2"
                  disabled={!!isDownloading}
                >
                  {isDownloading === 'appendix' ? <ArrowPathIcon className="w-4 h-4 animate-spin" /> : <ArrowDownTrayIcon className="w-4 h-4" />}
                  <span>{isDownloading === 'appendix' ? 'Downloading...' : 'Download Appendix'}</span>
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
          <div className="prose prose-lg max-w-none max-h-[70vh] overflow-y-auto">
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
          <div className="prose max-w-none max-h-[60vh] overflow-y-auto">
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
