import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';

export type HazardState = 'NOMINAL' | 'MONITOR' | 'ATTENUATE' | 'CRITICAL_SAFE_MODE';

export interface SatelliteHazardEntry {
  satelliteId: string;
  constellation: 'NavIC' | 'GSAT' | 'EOS/Cartosat' | 'INSAT';
  orbitType: 'GEO' | 'GSO' | 'SSO / LEO';
  altitudeKm: number;
  sepProtonFluxPfu: number;       // pfu (>10 MeV)
  doseRateRadPerHr: number;
  seuProbabilityPerHour: number;  // Single-Event Upset rate
  hazardState: HazardState;
  recommendedAction: string;
}

interface ConstellationHazardMatrixProps {
  satellites?: SatelliteHazardEntry[];
  onTriggerSafeMode?: (sat: SatelliteHazardEntry) => void;
}

export const DEFAULT_CONSTELLATION_FLEET: SatelliteHazardEntry[] = [
  {
    satelliteId: 'NavIC-1I (IRNSS-1I)',
    constellation: 'NavIC',
    orbitType: 'GSO',
    altitudeKm: 35786,
    sepProtonFluxPfu: 485.0,
    doseRateRadPerHr: 1.42,
    seuProbabilityPerHour: 0.18,
    hazardState: 'ATTENUATE',
    recommendedAction: 'Orient solar arrays along magnetic vector; bias star tracker thresholds',
  },
  {
    satelliteId: 'GSAT-7A (Angry Bird)',
    constellation: 'GSAT',
    orbitType: 'GEO',
    altitudeKm: 35786,
    sepProtonFluxPfu: 620.0,
    doseRateRadPerHr: 1.85,
    seuProbabilityPerHour: 0.24,
    hazardState: 'CRITICAL_SAFE_MODE',
    recommendedAction: 'Disable Ku-band high-voltage TWTA amplifiers; latch redundant gyros',
  },
  {
    satelliteId: 'GSAT-24 (CMS-02)',
    constellation: 'GSAT',
    orbitType: 'GEO',
    altitudeKm: 35786,
    sepProtonFluxPfu: 310.0,
    doseRateRadPerHr: 0.95,
    seuProbabilityPerHour: 0.08,
    hazardState: 'MONITOR',
    recommendedAction: 'Maintain active telemetry downlink monitoring; log single-bit EDAC events',
  },
  {
    satelliteId: 'EOS-04 (Radar Imaging)',
    constellation: 'EOS/Cartosat',
    orbitType: 'SSO / LEO',
    altitudeKm: 529,
    sepProtonFluxPfu: 45.0,
    doseRateRadPerHr: 0.12,
    seuProbabilityPerHour: 0.01,
    hazardState: 'NOMINAL',
    recommendedAction: 'Nominal SAR imaging cycle; geomagnetic cutoff shielding active',
  },
];

export const ConstellationHazardMatrix: React.FC<ConstellationHazardMatrixProps> = ({
  satellites = DEFAULT_CONSTELLATION_FLEET,
  onTriggerSafeMode,
}) => {
  const [selectedSat, setSelectedSat] = useState<SatelliteHazardEntry | null>(null);

  const getHazardBadge = (state: HazardState) => {
    switch (state) {
      case 'CRITICAL_SAFE_MODE':
        return (
          <Badge className="bg-red-950/80 text-red-300 border border-red-700 animate-pulse font-mono">
            SAFE-MODE ADVISORY
          </Badge>
        );
      case 'ATTENUATE':
        return (
          <Badge className="bg-amber-950/80 text-amber-300 border border-amber-700 font-mono">
            ATTENUATE PAYLOAD
          </Badge>
        );
      case 'MONITOR':
        return (
          <Badge className="bg-yellow-950/70 text-yellow-300 border border-yellow-700 font-mono">
            ELEVATED MONITOR
          </Badge>
        );
      case 'NOMINAL':
        return (
          <Badge className="bg-emerald-950/70 text-emerald-300 border border-emerald-700 font-mono">
            NOMINAL
          </Badge>
        );
    }
  };

  return (
    <Card className="bg-slate-950/90 border-slate-800 text-slate-100">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono tracking-wider text-cyan-400 uppercase">
            Indian Satellite Fleet Radiation & SEU Hazard Matrix
          </CardTitle>
          <span className="text-xs font-mono text-slate-400">
            ISRO Telemetry, Tracking and Command Network (ISTRAC) Feed
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Space Asset</th>
                <th className="py-2.5 px-3">Orbit</th>
                <th className="py-2.5 px-3">Proton Flux (&gt;10 MeV)</th>
                <th className="py-2.5 px-3">Dose Rate (rad/hr)</th>
                <th className="py-2.5 px-3">SEU Risk / hr</th>
                <th className="py-2.5 px-3">Advisory Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {satellites.map((sat) => {
                const isSelected = selectedSat?.satelliteId === sat.satelliteId;
                return (
                  <tr
                    key={sat.satelliteId}
                    onClick={() => {
                      setSelectedSat(sat);
                      if (onTriggerSafeMode && sat.hazardState === 'CRITICAL_SAFE_MODE') {
                        onTriggerSafeMode(sat);
                      }
                    }}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-cyan-950/40 text-cyan-200' : 'hover:bg-slate-900/60'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-semibold text-slate-200">
                      {sat.satelliteId}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {sat.orbitType} ({sat.altitudeKm.toLocaleString()} km)
                    </td>
                    <td className="py-2.5 px-3 text-amber-400 font-medium">
                      {sat.sepProtonFluxPfu.toFixed(1)} pfu
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {sat.doseRateRadPerHr.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`font-semibold ${
                          sat.seuProbabilityPerHour > 0.15
                            ? 'text-red-400'
                            : sat.seuProbabilityPerHour > 0.05
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }`}
                      >
                        {(sat.seuProbabilityPerHour * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {getHazardBadge(sat.hazardState)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};
