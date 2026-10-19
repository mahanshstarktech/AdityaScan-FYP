import React from 'react';
import { Gauge } from 'lucide-react';

export interface ConformalRiskBand {
  horizon: string;
  pMean: number;
  pLow: number;
  pHigh: number;
  riskLevel: 'LOW' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
}

export const ProbabilityGauges: React.FC = () => {
  const horizons: ConformalRiskBand[] = [
    { horizon: '30m', pMean: 0.18, pLow: 0.12, pHigh: 0.26, riskLevel: 'LOW' },
    { horizon: '1h', pMean: 0.34, pLow: 0.24, pHigh: 0.45, riskLevel: 'ELEVATED' },
    { horizon: '3h', pMean: 0.62, pLow: 0.48, pHigh: 0.74, riskLevel: 'HIGH' },
    { horizon: '6h', pMean: 0.78, pLow: 0.65, pHigh: 0.88, riskLevel: 'CRITICAL' },
    { horizon: '24h', pMean: 0.85, pLow: 0.72, pHigh: 0.94, riskLevel: 'CRITICAL' },
  ];

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-950/70 p-5 backdrop-blur-xl">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2.5">
          <Gauge className="h-5 w-5 text-cyan-400" />
          <h3 className="text-sm font-semibold tracking-wide text-slate-100 uppercase">
            Multi-Horizon Flare Probability Gauges
          </h3>
        </div>
      </div>
      <div className="mt-5 grid grid-cols-2 md:grid-cols-5 gap-4">
        {horizons.map((item) => (
          <div key={item.horizon} className="flex flex-col items-center rounded-lg border border-slate-800 bg-slate-900/40 p-4">
            <span className="text-xs font-mono font-bold text-slate-300">{item.horizon} Horizon</span>
            <span className="text-2xl font-mono font-bold text-rose-400 mt-2">{(item.pMean * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};
