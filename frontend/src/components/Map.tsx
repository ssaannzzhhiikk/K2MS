// /src/components/Map.tsx
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';
import 'leaflet/dist/leaflet.css';

// Fix for default markers
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

// Define the Geotrack interface
interface Geotrack {
  lat: number;
  lng: number;
  routeId: string;
}

// Component to handle heatmap layer
const HeatmapLayer: React.FC<{ points: [number, number, number][] }> = ({ points }) => {
  const map = useMap();

  useEffect(() => {
    if (!map || points.length === 0) return;

    // Create heatmap layer using leaflet.heat
    const heatLayer = (L as any).heatLayer(points, {
      radius: 25,
      blur: 15,
      maxZoom: 17,
      gradient: {
        0.4: 'blue',
        0.65: 'lime',
        1: 'red'
      }
    });

    // Add to map
    map.addLayer(heatLayer);

    // Cleanup function
    return () => {
      map.removeLayer(heatLayer);
    };
  }, [map, points]);

  return null;
};

// You'll need to implement this function or import it from somewhere
const getGeotracks = async (): Promise<Geotrack[]> => {
  // This is a placeholder - replace with your actual API call
  return [
    { lat: 51.505, lng: -0.09, routeId: "route1" },
    { lat: 51.51, lng: -0.1, routeId: "route2" },
    { lat: 51.515, lng: -0.08, routeId: "route3" },
    { lat: 51.52, lng: -0.11, routeId: "route4" },
    { lat: 51.507, lng: -0.095, routeId: "route5" },
    // Add your actual data fetching logic here
  ];
};

const Map: React.FC = () => {
  const [geotracks, setGeotracks] = useState<Geotrack[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const data = await getGeotracks();
      setGeotracks(data);
    };
    fetchData();
  }, []);

  // Convert data to heatmap format [lat, lng, intensity]
  const heatmapData: [number, number, number][] = geotracks.map(track => [
    track.lat, 
    track.lng, 
    1 // intensity value - you can adjust this based on your data
  ]);

  return (
    <MapContainer center={[51.505, -0.09]} zoom={13} style={{ height: '100vh' }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <HeatmapLayer points={heatmapData} />
      {geotracks.map((track, index) => (
        <Marker position={[track.lat, track.lng]} key={index}>
          <Popup>
            <span>Route ID: {track.routeId}</span>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default Map;