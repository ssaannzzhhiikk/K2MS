// /src/pages/Dashboard.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const Dashboard: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Popular Routes */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="font-semibold text-lg">Popular Routes</h3>
          <p className="mt-2">Explore the most popular routes from the data.</p>
          {/* You could add charts, graphs, or heatmaps for popular routes */}
        </div>

        {/* Route Optimization */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="font-semibold text-lg">Route Optimization</h3>
          <p className="mt-2">Analyze and optimize driver routes to improve efficiency.</p>
        </div>

        {/* Safety Analysis */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="font-semibold text-lg">Safety Analysis</h3>
          <p className="mt-2">Identify risky routes and take actions for safety.</p>
        </div>
      </div>

      {/* Link to Map Page */}
      <div className="mt-6">
        <Link to="/map" className="text-blue-500 hover:text-blue-700">
          View Geotrack Map
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;
