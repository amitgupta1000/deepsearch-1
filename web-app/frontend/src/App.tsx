
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import ResearchForm from './components/ResearchForm';
import ResultsDisplay from './components/ResultsDisplay';
import WelcomeSection from './components/WelcomeSection';
import ActivityLog from './components/ActivityLog';
import Footer from './components/Footer';

import { ResearchProvider } from './context/ResearchContext';

function App() {
  return (
    <ResearchProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 flex flex-col">
          <Header />
          <main className="flex-1 container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={
                <div className="space-y-12">
                  <WelcomeSection />
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div className="lg:col-span-2">
                      <ResearchForm />
                      <ActivityLog />
                    </div>
                    <div className="lg:col-span-1">
                      <div className="sticky top-8">
                        <div className="card p-6">
                          <h3 className="text-lg font-semibold text-gray-900 mb-4">Research Tips</h3>
                          <ul className="space-y-2 text-sm text-gray-600">
                            <li>• Be specific in your research query</li>
                            <li>• Choose the appropriate report type</li>
                            <li>• Detailed reports provide comprehensive analysis</li>
                            <li>• Concise reports offer quick insights</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                  <ResultsDisplay />
                </div>
              } />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </ResearchProvider>
  );
}

export default App;
