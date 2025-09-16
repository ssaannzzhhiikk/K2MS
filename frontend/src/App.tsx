// /src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import MapPage from './pages/MapPage';
import AnalyticsPage from './pages/AnalyticsPage';
import './App.css'

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <nav className="p-4 bg-blue-600 text-white">
          <ul className="flex space-x-4">
            <li><a href="/">Dashboard</a></li>
            <li><a href="/map">Map</a></li>
            <li><a href="/analytics">Analytics</a></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
