export interface StormScenario {
  id: string;
  name: string;
  peakFlux: number;
  flareClass: string;
  geomagneticKp: number;
  durationMinutes: number;
  description: string;
}

export const MAY_2024_STORM: StormScenario = {
  id: "may-2024-gannon",
  name: "May 2024 Gannon Superstorm",
  peakFlux: 5.8e-4, // X5.8 flare
  flareClass: "X5.8",
  geomagneticKp: 9.0,
  durationMinutes: 180,
  description: "Extreme flare storm analogous to the historic May 10-12, 2024 event."
};

export function injectStormTelemetry(baseFlux: number, progressRatio: number): number {
  if (progressRatio <= 0 || progressRatio >= 1) return baseFlux;
  // Gaussian profile of flare eruption
  const gaussian = Math.exp(-0.5 * Math.pow((progressRatio - 0.4) / 0.15, 2));
  return baseFlux + (MAY_2024_STORM.peakFlux - baseFlux) * gaussian;
}
