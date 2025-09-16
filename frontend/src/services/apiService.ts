export interface Geotrack {
  routeId: string;
  lat: number;
  lng: number;
  timestamp: string;
}

export const getGeotracks = async (): Promise<Geotrack[]> => {
  try {
    const response = await fetch('/api/geotracks');
    if (!response.ok) throw new Error('Failed to fetch geotracks');
    const data: Geotrack[] = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching geotracks:', error);
    return [];
  }
};
