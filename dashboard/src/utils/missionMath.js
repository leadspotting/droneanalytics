const METRES_PER_LAT_DEG = 111320.0;

export function computeRouteDistance(waypoints) {
  if (!waypoints || waypoints.length < 2) return 0;

  let total = 0;
  for (let index = 0; index < waypoints.length - 1; index += 1) {
    const current = waypoints[index];
    const next = waypoints[index + 1];
    const midLatRad = (Math.PI / 180) * ((current.lat + next.lat) / 2);
    const dy = (next.lat - current.lat) * METRES_PER_LAT_DEG;
    const dx = (next.lon - current.lon) * METRES_PER_LAT_DEG * Math.cos(midLatRad);
    total += Math.sqrt(dx * dx + dy * dy);
  }

  return total;
}

export function computeFlightTime(waypoints) {
  return Math.round((computeRouteDistance(waypoints) / 10.0) * 10) / 10;
}

export function computeFeasibility(waypoints, drone) {
  if (!waypoints || !drone) return null;

  const routeDistanceMeters = computeRouteDistance(waypoints);
  if (routeDistanceMeters <= 0) {
    return {
      feasible: true,
      estimated_time_min: 0,
      safe_flight_time_min: drone.safe_flight_time_min,
      distance_km: 0,
      battery_usage_pct: 0,
      reason: '',
    };
  }

  const cruiseSpeedMetersPerSecond = (drone.cruise_speed_kmh * 1000) / 3600;
  const estimatedMinutes = routeDistanceMeters / cruiseSpeedMetersPerSecond / 60;
  const safeMinutes = drone.safe_flight_time_min;
  const feasible = estimatedMinutes <= safeMinutes;
  const batteryUsagePercent = Math.min(
    safeMinutes > 0 ? (estimatedMinutes / safeMinutes) * 100 : 200,
    200,
  );

  return {
    feasible,
    estimated_time_min: Math.round(estimatedMinutes * 10) / 10,
    safe_flight_time_min: Math.round(safeMinutes * 10) / 10,
    distance_km: Math.round(routeDistanceMeters / 10) / 100,
    battery_usage_pct: Math.round(batteryUsagePercent * 10) / 10,
    reason: feasible
      ? ''
      : `Mission needs ~${estimatedMinutes.toFixed(1)} min but safe limit for ${drone.model} is ${safeMinutes.toFixed(1)} min`,
  };
}
