import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';

export interface ArcadeLink {
  sourceAr: string;
  targetAr: string;
  angularSeparationDeg: number;
  magneticCouplingCoeff: number; // 0 to 1
  sharedFluxMaxwell: number;     // in 10^21 Mx
  sympatheticRiskScore: number;  // 0 to 100
  status: 'active_arcade' | 'trans-equatorial' | 'quiescent';
}

interface CoronalArcadeMatrixProps {
  links?: ArcadeLink[];
  onSelectLink?: (link: ArcadeLink) => void;
}

export const DEFAULT_ARCADE_LINKS: ArcadeLink[] = [
  {
    sourceAr: 'AR13664',
    targetAr: 'AR13668',
    angularSeparationDeg: 14.2,
    magneticCouplingCoeff: 0.88,
    sharedFluxMaxwell: 2.45,
    sympatheticRiskScore: 92.4,
    status: 'trans-equatorial',
  },
  {
    sourceAr: 'AR13664',
    targetAr: 'AR13663',
    angularSeparationDeg: 19.8,
    magneticCouplingCoeff: 0.64,
    sharedFluxMaxwell: 1.12,
    sympatheticRiskScore: 68.0,
    status: 'active_arcade',
  },
  {
    sourceAr: 'AR13668',
    targetAr: 'AR13670',
    angularSeparationDeg: 28.5,
    magneticCouplingCoeff: 0.31,
    sharedFluxMaxwell: 0.42,
    sympatheticRiskScore: 35.5,
    status: 'quiescent',
  },
];

export const CoronalArcadeMatrix: React.FC<CoronalArcadeMatrixProps> = ({
  links = DEFAULT_ARCADE_LINKS,
  onSelectLink,
}) => {
  const [selectedLink, setSelectedLink] = useState<ArcadeLink | null>(null);

  const handleRowClick = (link: ArcadeLink) => {
    setSelectedLink(link);
    if (onSelectLink) {
      onSelectLink(link);
    }
  };

  const getStatusBadge = (status: ArcadeLink['status']) => {
    switch (status) {
      case 'trans-equatorial':
        return <Badge className="bg-red-900/60 text-red-300 border border-red-700">Trans-Equatorial</Badge>;
      case 'active_arcade':
        return <Badge className="bg-amber-900/60 text-amber-300 border border-amber-700">Active Arcade</Badge>;
      case 'quiescent':
        return <Badge className="bg-slate-800 text-slate-300 border border-slate-700">Quiescent</Badge>;
    }
  };

  return (
    <Card className="bg-slate-950/80 border-slate-800 text-slate-100">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono tracking-wider text-amber-400 uppercase">
            Coronal Loop & Sympathetic Arcade Matrix
          </CardTitle>
          <span className="text-xs font-mono text-slate-400">
            SUIT 279nm / VELC Field-of-View
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 px-3">Node Pair</th>
                <th className="py-2 px-3">Separation (deg)</th>
                <th className="py-2 px-3">Coupling (r_ij)</th>
                <th className="py-2 px-3">Shared Flux (10²¹ Mx)</th>
                <th className="py-2 px-3">Sympathetic Risk</th>
                <th className="py-2 px-3">Arcade Regime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {links.map((link) => {
                const isSelected =
                  selectedLink?.sourceAr === link.sourceAr &&
                  selectedLink?.targetAr === link.targetAr;
                return (
                  <tr
                    key={`${link.sourceAr}-${link.targetAr}`}
                    onClick={() => handleRowClick(link)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-amber-950/40 text-amber-200' : 'hover:bg-slate-900/60'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-semibold text-slate-200">
                      {link.sourceAr} ↔ {link.targetAr}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {link.angularSeparationDeg.toFixed(1)}°
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-amber-400 h-1.5 rounded-full"
                            style={{ width: `${link.magneticCouplingCoeff * 100}%` }}
                          />
                        </div>
                        <span className="text-amber-400 font-medium">
                          {(link.magneticCouplingCoeff * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {link.sharedFluxMaxwell.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`font-semibold ${
                          link.sympatheticRiskScore > 80
                            ? 'text-red-400'
                            : link.sympatheticRiskScore > 50
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }`}
                      >
                        {link.sympatheticRiskScore.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      {getStatusBadge(link.status)}
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
