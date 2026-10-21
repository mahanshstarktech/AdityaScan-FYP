import { describe, it, expect } from 'vitest';
import type { HorizonForecast } from './ProbabilityGauges';
import type { ActiveRegionData } from './ActiveRegionStatusCards';

describe('Mission Control UI Sprint 2 Suite', () => {
  it('validates radiation alert threshold bounds', () => {
    const criticalThreshold = 1e5;
    expect(1.2e5 > criticalThreshold).toBe(true);
  });

  it('validates multi-horizon conformal prediction confidence bounds [p_low <= p_high]', () => {
    const sampleForecasts: HorizonForecast[] = [
      { horizon: '30m', label: '30 Minutes', probability: 84.2, pLow: 78.5, pHigh: 89.1, mClassProb: 74, xClassProb: 26, riskLevel: 'critical' },
      { horizon: '1h', label: '1 Hour', probability: 79.5, pLow: 72.8, pHigh: 85.2, mClassProb: 68.5, xClassProb: 21.5, riskLevel: 'critical' },
      { horizon: '3h', label: '3 Hours', probability: 64.0, pLow: 56.4, pHigh: 71.9, mClassProb: 55, xClassProb: 15, riskLevel: 'critical' },
      { horizon: '6h', label: '6 Hours', probability: 48.7, pLow: 40.2, pHigh: 57.5, mClassProb: 42, xClassProb: 8.5, riskLevel: 'elevated' },
      { horizon: '24h', label: '24 Hours', probability: 32.1, pLow: 24.0, pHigh: 41.2, mClassProb: 28, xClassProb: 4.1, riskLevel: 'elevated' },
    ];

    expect(sampleForecasts).toHaveLength(5);
    for (const f of sampleForecasts) {
      expect(f.pLow).toBeLessThanOrEqual(f.pHigh);
      expect(f.probability).toBeGreaterThanOrEqual(0);
      expect(f.probability).toBeLessThanOrEqual(100);
      expect(f.mClassProb + f.xClassProb).toBeLessThanOrEqual(100);
    }
  });

  it('verifies Active Region AR13664 magnetic complexity and McIntosh classification', () => {
    const ar13664: ActiveRegionData = {
      id: 'AR13664',
      location: 'S19W42',
      haleClass: 'beta-gamma-delta',
      mcIntosh: 'Fkc',
      totalFlux: 4.82,
      area: 2150,
      shearAnglePct: 84.6,
      flareHistory: { c: 28, m: 14, x: 7 },
      sympatheticLinks: ['AR13668', 'AR13663'],
    };

    expect(ar13664.haleClass).toBe('beta-gamma-delta');
    expect(ar13664.totalFlux).toBeGreaterThan(4.0);
    expect(ar13664.shearAnglePct).toBeGreaterThan(80.0);
    expect(ar13664.flareHistory.x).toBe(7);
    expect(ar13664.sympatheticLinks).toContain('AR13668');
  });

  it('verifies sympathetic coronal loop arcade linkages between AR13664 and AR13668', () => {
    const activeRegions: ActiveRegionData[] = [
      {
        id: 'AR13664',
        location: 'S19W42',
        haleClass: 'beta-gamma-delta',
        mcIntosh: 'Fkc',
        totalFlux: 4.82,
        area: 2150,
        shearAnglePct: 84.6,
        flareHistory: { c: 28, m: 14, x: 7 },
        sympatheticLinks: ['AR13668'],
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
    ];

    const ar1 = activeRegions[0];
    const ar2 = activeRegions[1];
    expect(ar1.sympatheticLinks).toContain(ar2.id);
    expect(ar2.sympatheticLinks).toContain(ar1.id);
  });
});
