import React from 'react';

export interface ProbabilityGaugesProps {
  probabilities?: Record<string, number>;
}

export const ProbabilityGauges: React.FC<ProbabilityGaugesProps> = ({ probabilities }) => {
  const horizons = [
    { key: "30m", label: "30 min", defaultVal: "18%" },
    { key: "1h",  label: "1 hour", defaultVal: "34%" },
    { key: "3h",  label: "3 hour", defaultVal: "62%" },
    { key: "6h",  label: "6 hour", defaultVal: "78%" },
    { key: "24h", label: "24 hour", defaultVal: "85%" },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5">
      <h3 className="text-sm font-semibold text-slate-200">Multi-Horizon Forecast Risk</h3>
      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-4">
        {horizons.map(({ key, label, defaultVal }) => (
          <div key={key} className="flex flex-col items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
            <span className="text-xs text-slate-400 uppercase">{label}</span>
            <span className="text-lg font-mono font-bold text-cyan-400 mt-1">
              {probabilities?.[key] ? `${(probabilities[key] * 100).toFixed(0)}%` : defaultVal}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
