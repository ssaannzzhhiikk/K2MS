import React from 'react';
import Map from '../components/Map';

const MapPage: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold">Map View</h1>
      <p className="mt-4">Here, you can visualize the geotracks on the map and view the heatmaps of popular routes.</p>
      {/* Render the Map component */}
      <Map />
    </div>
  );
};

export default MapPage;
