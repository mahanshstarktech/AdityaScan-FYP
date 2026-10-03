import React from 'react';
import { Shield, AlertTriangle, Radio } from 'lucide-react';

interface RadiationBannerProps {
  fluxLevel: 'NOMINAL' | 'ELEVATED' | 'CRITICAL';
  electronFluxGeo: number; // particles / (cm² s sr)
  affectedConstellations: string[];
}

export const RadiationBanner: React.FC<RadiationBannerProps> = ({
  fluxLevel,
  electronFluxGeo,
  affectedConstellations,
}) => {
  if (fluxLevel === 'NOMINAL') return null;

  const isCritical = fluxLevel === 'CRITICAL';

  return (
    <div className={`w-full border-b px-4 py-2.5 backdrop-blur-md transition-all ${
      isCritical ? 'border-rose-500/50 bg-rose-950/40 text-rose-200' : 'border-amber-500/50 bg-amber-950/40 text-amber-200'
    }`}>
      <div className="mx-auto flex max-w-7xl items-center justify-between text-xs">
        <div className="flex items-center gap-3">
          <span className={`flex h-2 w-2 rounded-full ${isCritical ? 'bg-rose-500 animate-ping' : 'bg-amber-400'}`} />
          <span className="font-bold tracking-wide uppercase">
            {isCritical ? 'CRITICAL RADIATION ALERT' : 'ELEVATED RADIATION ADVISORY'}
          </span>
          <span className="text-slate-300">
            Outer-belt GEO relativistic electron flux: <strong className="font-mono">{electronFluxGeo.toExponential(2)}</strong>
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <span className="text-slate-400">Impacted Assets:</span>
          {affectedConstellations.map(c => (
            <span key={c} className="rounded bg-slate-900/80 px-2 py-0.5 font-mono text-cyan-300 border border-slate-700">
              {c}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};
