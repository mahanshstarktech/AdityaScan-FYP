import React from 'react';

export interface HorizonForecast {
  horizon: '30m' | '1h' | '3h' | '6h' | '24h';
  label: string;
  probability: number; // 0 to 100
  pLow: number; // 90% conformal lower bound
  pHigh: number; // 90% conformal upper bound
  mClassProb: number;
  xClassProb: number;
  riskLevel: 'nominal' | 'elevated' | 'critical';
}

const DEFAULT_FORECASTS: HorizonForecast[] = [
  {
    horizon: '30m',
    label: '30 Minutes',
    probability: 84.2,
    pLow: 78.5,
    pHigh: 89.1,
    mClassProb: 74.0,
    xClassProb: 26.0,
    riskLevel: 'critical',
  },
  {
    horizon: '1h',
    label: '1 Hour',
    probability: 79.5,
    pLow: 72.8,
    pHigh: 85.2,
    mClassProb: 68.5,
    xClassProb: 21.5,
    riskLevel: 'critical',
  },
  {
    horizon: '3h',
    label: '3 Hours',
    probability: 64.0,
    pLow: 56.4,
    pHigh: 71.9,
    mClassProb: 55.0,
    xClassProb: 15.0,
    riskLevel: 'critical',
  },
  {
    horizon: '6h',
    label: '6 Hours',
    probability: 48.7,
    pLow: 40.2,
    pHigh: 57.5,
    mClassProb: 42.0,
    xClassProb: 8.5,
    riskLevel: 'elevated',
  },
  {
    horizon: '24h',
    label: '24 Hours',
    probability: 32.1,
    pLow: 24.0,
    pHigh: 41.2,
    mClassProb: 28.0,
    xClassProb: 4.1,
    riskLevel: 'elevated',
  },
];

const COLOR_MAP = {
  critical: {
    stroke: '#F43F5E', // Rose 500
    glow: 'rgba(244, 63, 94, 0.4)',
    text: 'text-rose-400',
    bg: 'bg-rose-950/40',
    border: 'border-rose-500/30',
  },
  elevated: {
    stroke: '#F59E0B', // Amber 500
    glow: 'rgba(245, 158, 11, 0.4)',
    text: 'text-amber-400',
    bg: 'bg-amber-950/40',
    border: 'border-amber-500/30',
  },
  nominal: {
    stroke: '#10B981', // Emerald 500
    glow: 'rgba(16, 185, 129, 0.3)',
    text: 'text-emerald-400',
    bg: 'bg-emerald-950/40',
    border: 'border-emerald-500/30',
  },
};

interface ProbabilityGaugesProps {
  forecasts?: HorizonForecast[];
  onSelectHorizon?: (horizon: HorizonForecast) => void;
}

export const ProbabilityGauges: React.FC<ProbabilityGaugesProps> = ({
  forecasts = DEFAULT_FORECASTS,
  onSelectHorizon,
}) => {
  const radius = 34;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
          <h3 className="text-xs font-semibold tracking-wider text-slate-300 uppercase">
            Multi-Horizon Flare Probability Gauges & Conformal Uncertainty (90% CI)
          </h3>
        </div>
        <span className="text-[10px] text-slate-500 font-mono">Model: SpatialGNN + TCN Cross-Attention</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {forecasts.map((f) => {
          const config = COLOR_MAP[f.riskLevel];
          const strokeDashoffset = circumference - (f.probability / 100) * circumference;

          return (
            <div
              key={f.horizon}
              onClick={() => onSelectHorizon?.(f)}
              className={`group relative flex flex-col items-center rounded-xl border ${config.border} bg-slate-950/60 p-3 backdrop-blur-md transition-all duration-200 hover:border-slate-600 hover:bg-slate-900/60 hover:shadow-lg cursor-pointer`}
            >
              {/* Top Horizon Label */}
              <div className="text-[11px] font-mono font-medium text-slate-400 mb-2">
                Horizon: <span className="text-slate-200 font-semibold">{f.label}</span>
              </div>

              {/* Radial SVG Dial */}
              <div className="relative h-24 w-24 flex items-center justify-center">
                <svg className="h-full w-full -rotate-90 transform" viewBox="0 0 80 80">
                  {/* Background Track */}
                  <circle
                    cx="40"
                    cy="40"
                    r={radius}
                    className="stroke-slate-800"
                    strokeWidth="6"
                    fill="transparent"
                  />
                  {/* Active Value Arc */}
                  <circle
                    cx="40"
                    cy="40"
                    r={radius}
                    stroke={config.stroke}
                    strokeWidth="6"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    fill="transparent"
                    style={{
                      transition: 'stroke-dashoffset 0.8s ease-out',
                      filter: `drop-shadow(0 0 6px ${config.glow})`,
                    }}
                  />
                </svg>

                {/* Dial Center Content */}
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="font-mono text-base font-bold text-white tracking-tight">
                    {f.probability.toFixed(1)}%
                  </span>
                  <span className={`text-[9px] font-mono font-semibold uppercase ${config.text}`}>
                    {f.riskLevel}
                  </span>
                </div>
              </div>

              {/* Conformal Uncertainty Interval */}
              <div className="mt-2 w-full text-center border-t border-slate-900 pt-1.5">
                <div className="text-[9px] text-slate-500 font-mono">90% Coverage Band:</div>
                <div className="font-mono text-[10px] font-semibold text-slate-300">
                  [{f.pLow.toFixed(1)}% – {f.pHigh.toFixed(1)}%]
                </div>
              </div>

              {/* M & X Class Probabilities */}
              <div className="mt-1.5 flex items-center justify-between w-full px-1 text-[9px] font-mono text-slate-400">
                <span>M: {f.mClassProb.toFixed(0)}%</span>
                <span>X: {f.xClassProb.toFixed(1)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
export default ProbabilityGauges;
