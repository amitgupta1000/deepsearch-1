import React from 'react';
import { FileText, Clock, ExternalLink, Download, AlertCircle, Loader2, CheckCircle, Copy, Share2 } from 'lucide-react';
import { useResearch } from '../context/ResearchContext';

const ResultsDisplay: React.FC = () => {
  const { state, clearResults } = useResearch();
  const { isLoading, result, error } = state;

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      // You could add a toast notification here
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const shareResults = async () => {
    if (navigator.share && result) {
      try {
        await navigator.share({
          title: 'INTELLISEARCH Research Report',
          text: result.report.substring(0, 100) + '...',
          url: window.location.href,
        });
      } catch (err) {
        console.error('Error sharing:', err);
      }
    }
  };

  if (isLoading) {
    return (
      <div className="card p-8 shadow-lg">
        <div className="flex flex-col items-center space-y-6">
          <div className="relative">
            <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
            <div className="absolute inset-0 w-12 h-12 border-4 border-primary-200 rounded-full"></div>
          </div>
          <div className="text-center space-y-2">
            <h3 className="text-xl font-semibold text-gray-900">Research in Progress</h3>
            <p className="text-gray-600">Our AI is analyzing sources and generating your report...</p>
            <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 border-red-200 bg-red-50 shadow-lg">
        <div className="flex items-start space-x-4">
          <AlertCircle className="w-6 h-6 text-red-600 mt-1 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="text-xl font-semibold text-red-900 mb-2">Research Error</h3>
            <p className="text-red-700 mb-4">{error}</p>
            <div className="flex space-x-3">
              <button
                onClick={clearResults}
                className="btn-primary bg-red-600 hover:bg-red-700"
              >
                Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="btn-secondary"
              >
                Refresh Page
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const downloadReport = () => {
    const blob = new Blob([result.report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `intellisearch-report-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadReportAs = async (format: 'txt' | 'pdf') => {
    if (!result.sessionId) {
      // Fallback to local download for results without session ID
      downloadReport();
      return;
    }

    try {
      const response = await fetch(`/api/research/${result.sessionId}/download?format=${format}`);
      
      if (!response.ok) {
        throw new Error(`Failed to download ${format.toUpperCase()} file`);
      }
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `intellisearch-report-${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error(`Error downloading ${format.toUpperCase()} file:`, error);
      // Fallback to local text download
      if (format === 'txt') {
        downloadReport();
      } else {
        alert(`Failed to download ${format.toUpperCase()} file. Please try again.`);
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* Success Header */}
      <div className="card p-6 bg-gradient-to-r from-green-50 to-blue-50 border-green-200 shadow-lg">
        <div className="flex items-center space-x-3 mb-4">
          <CheckCircle className="w-8 h-8 text-green-600" />
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Research Complete!</h2>
            <p className="text-gray-600">Your comprehensive report is ready</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary-600">{result.wordCount}</div>
            <div className="text-sm text-gray-600">Words</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{result.sources?.length || 0}</div>
            <div className="text-sm text-gray-600">Sources</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{result.citations?.length || 0}</div>
            <div className="text-sm text-gray-600">Citations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {Math.ceil(result.wordCount / 250)}
            </div>
            <div className="text-sm text-gray-600">Min Read</div>
          </div>
        </div>
      </div>

      {/* Report Header */}
      <div className="card p-6 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between space-y-4 md:space-y-0">
          <div className="flex items-center space-x-3">
            <FileText className="w-6 h-6 text-primary-600" />
            <div>
              <h3 className="text-xl font-semibold text-gray-900">Research Report</h3>
              <div className="flex items-center space-x-4 text-sm text-gray-600">
                <span className="flex items-center space-x-1">
                  <Clock className="w-4 h-4" />
                  <span>{new Date(result.timestamp).toLocaleString()}</span>
                </span>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => copyToClipboard(result.report)}
              className="btn-secondary flex items-center space-x-1"
            >
              <Copy className="w-4 h-4" />
              <span>Copy</span>
            </button>
            {'share' in navigator && (
              <button
                onClick={shareResults}
                className="btn-secondary flex items-center space-x-1"
              >
                <Share2 className="w-4 h-4" />
                <span>Share</span>
              </button>
            )}
            <div className="flex items-center space-x-1">
              <button
                onClick={() => downloadReportAs('txt')}
                className="btn-secondary flex items-center space-x-1"
              >
                <FileText className="w-4 h-4" />
                <span>Download TXT</span>
              </button>
              <button
                onClick={() => downloadReportAs('pdf')}
                className="btn-secondary flex items-center space-x-1"
              >
                <Download className="w-4 h-4" />
                <span>Download PDF</span>
              </button>
            </div>
            <button
              onClick={clearResults}
              className="btn-primary"
            >
              New Research
            </button>
          </div>
        </div>
      </div>

      {/* Report Content */}
      <div className="card p-8 shadow-lg">
        <div className="prose prose-lg max-w-none">
          <div className="whitespace-pre-wrap text-gray-900 leading-relaxed font-medium">
            {result.report}
          </div>
        </div>
      </div>

      {/* Citations */}
      {result.citations && result.citations.length > 0 && (
        <div className="card p-6 shadow-lg">
          <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
            <ExternalLink className="w-5 h-5" />
            <span>Citations & References</span>
            <span className="text-sm bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
              {result.citations.length} sources
            </span>
          </h3>
          <div className="grid gap-4">
            {result.citations.map((citation, index) => (
              <div key={index} className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white text-sm font-bold rounded-full flex items-center justify-center">
                  {index + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 break-words leading-relaxed">{citation}</p>
                </div>
                <button
                  onClick={() => copyToClipboard(citation)}
                  className="flex-shrink-0 p-1 text-gray-400 hover:text-gray-600 transition-colors"
                  title="Copy citation"
                >
                  <Copy className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sources */}
      {result.sources && result.sources.length > 0 && (
        <div className="card p-6 shadow-lg">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Source URLs</h3>
          <div className="grid gap-3">
            {result.sources.map((source, index) => (
              <a
                key={index}
                href={source}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-3 p-4 bg-gray-50 hover:bg-primary-50 rounded-lg transition-colors group"
              >
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-primary-600 flex-shrink-0" />
                <span className="text-sm text-primary-600 hover:text-primary-800 break-all flex-1">
                  {source}
                </span>
                <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                  Source {index + 1}
                </span>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;