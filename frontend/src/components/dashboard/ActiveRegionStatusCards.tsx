import React from 'react';

export interface ActiveRegionData {
  id: string; // e.g. "AR13664"
  location: string; // e.g. "S19W42"
  haleClass: 'alpha' | 'beta' | 'beta-gamma' | 'beta-gamma-delta';
  mcIntosh: string; // e.g. "Fkc"
  totalFlux: number; // 10^22 Mx
  area: number; // Millionths of solar hemisphere
  shearAnglePct: number; // % area with shear > 45 deg
  flareHistory: {
    c: number;
    m: number;
    x: number;
  };
  sympatheticLinks?: string[]; // Neighboring ARs with coronal magnetic bridges
}

const DEFAULT_ACTIVE_REGIONS: ActiveRegionData[] = [
  {
    id: 'AR13664',
    location: 'S19W42',
    haleClass: 'beta-gamma-delta',
    mcIntosh: 'Fkc',
    totalFlux: 4.82,
    area: 2150,
    shearAnglePct: 84.6,
    flareHistory: { c: 28, m: 14, x: 7 },
    sympatheticLinks: ['AR13668', 'AR13663'],
  },
  {
    id: 'AR13668',
    location: 'S15W10',
    haleClass: 'beta-gamma',
    mcIntosh: 'Dkc',
    totalFlux: 2.15,
    area: 980,
    shearAnglePct: 62.1,
    flareHistory: { c: 12, m: 4, x: 1 },
    sympatheticLinks: ['AR13664'],
  },
  {
    id: 'AR13663',
    location: 'N24E31',
    haleClass: 'beta-gamma-delta',
    mcIntosh: 'Ekc',
    totalFlux: 3.40,
    area: 1450,
    shearAnglePct: 76.8,
    flareHistory: { c: 19, m: 9, x: 3 },
    sympatheticLinks: ['AR13664'],
  },
];

const HALE_BADGES: Record<ActiveRegionData['haleClass'], { label: string; bg: string; text: string; border: string }> = {
  'beta-gamma-delta': {
    label: 'βγδ (Severe Complexity)',
    bg: 'bg-rose-950/70',
    text: 'text-rose-300',
    border: 'border-rose-500/40',
  },
  'beta-gamma': {
    label: 'βγ (Elevated)',
    bg: 'bg-amber-950/70',
    text: 'text-amber-300',
    border: 'border-amber-500/40',
  },
  beta: {
    label: 'β (Moderate)',
    bg: 'bg-yellow-950/70',
    text: 'text-yellow-300',
    border: 'border-yellow-500/40',
  },
  alpha: {
    label: 'α (Unipolar/Quiet)',
    bg: 'bg-emerald-950/70',
    text: 'text-emerald-300',
    border: 'border-emerald-500/40',
  },
};

interface ActiveRegionStatusCardsProps {
  regions?: ActiveRegionData[];
  onSelectRegion?: (region: ActiveRegionData) => void;
}

export const ActiveRegionStatusCards: React.FC<ActiveRegionStatusCardsProps> = ({
  regions = DEFAULT_ACTIVE_REGIONS,
  onSelectRegion,
}) => {
  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
            Active Solar Regions (SDO/HMI Magnetogram CEA Patches)
          </h3>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">Cadence: 12m | SHARP CEA Series</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {regions.map((ar) => {
          const badge = HALE_BADGES[ar.haleClass];
          return (
            <div
              key={ar.id}
              onClick={() => onSelectRegion?.(ar)}
              className="group relative rounded-xl border border-slate-800/80 bg-slate-950/60 p-3.5 backdrop-blur-md transition-all duration-200 hover:border-slate-700/80 hover:bg-slate-900/50 hover:shadow-lg hover:shadow-cyan-950/20 cursor-pointer"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-bold text-white group-hover:text-cyan-400 transition-colors">
                      {ar.id}
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">({ar.location})</span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-500">McIntosh: {ar.mcIntosh}</span>
                </div>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-medium border font-mono ${badge.bg} ${badge.text} ${badge.border}`}
                >
                  {badge.label}
                </span>
              </div>

              {/* Metrics Grid */}
              <div className="mt-3 grid grid-cols-3 gap-2 border-t border-slate-800/60 pt-2.5">
                <div>
                  <div className="text-[10px] text-slate-500">Magnetic Flux</div>
                  <div className="font-mono text-xs font-semibold text-slate-200">
                    {ar.totalFlux} <span className="text-[9px] text-slate-500">×10²² Mx</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">Area</div>
                  <div className="font-mono text-xs font-semibold text-slate-200">
                    {ar.area} <span className="text-[9px] text-slate-500">MSH</span>
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">Shear &gt;45°</div>
                  <div className="font-mono text-xs font-semibold text-rose-400">{ar.shearAnglePct}%</div>
                </div>
              </div>

              {/* Flare Counts & Coronal Loops */}
              <div className="mt-2.5 flex items-center justify-between text-[11px] border-t border-slate-900 pt-2">
                <div className="flex items-center gap-1.5 font-mono text-[10px]">
                  <span className="text-slate-500">Flares:</span>
                  <span className="text-amber-400 font-semibold">{ar.flareHistory.m}M</span>
                  <span className="text-rose-400 font-bold">{ar.flareHistory.x}X</span>
                  <span className="text-slate-400">{ar.flareHistory.c}C</span>
                </div>
                {ar.sympatheticLinks && ar.sympatheticLinks.length > 0 && (
                  <div className="flex items-center gap-1 text-[9px] text-cyan-400 font-mono">
                    <span className="h-1 w-1 rounded-full bg-cyan-400" />
                    <span>Loop: {ar.sympatheticLinks.join(', ')}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
export default ActiveRegionStatusCards;
