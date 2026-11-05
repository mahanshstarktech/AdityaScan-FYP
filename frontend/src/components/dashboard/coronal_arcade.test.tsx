import { describe, it, expect } from 'vitest';
import { DEFAULT_ARCADE_LINKS, type ArcadeLink } from './CoronalArcadeMatrix';

describe('CoronalArcadeMatrix Component Test Suite', () => {
  it('validates default active arcade connection links', () => {
    expect(DEFAULT_ARCADE_LINKS).toHaveLength(3);
    const primaryLink = DEFAULT_ARCADE_LINKS[0];
    expect(primaryLink.sourceAr).toBe('AR13664');
    expect(primaryLink.targetAr).toBe('AR13668');
    expect(primaryLink.magneticCouplingCoeff).toBeGreaterThan(0.8);
    expect(primaryLink.status).toBe('trans-equatorial');
  });

  it('verifies magnetic coupling coefficient bounds [0, 1]', () => {
    for (const link of DEFAULT_ARCADE_LINKS) {
      expect(link.magneticCouplingCoeff).toBeGreaterThanOrEqual(0.0);
      expect(link.magneticCouplingCoeff).toBeLessThanOrEqual(1.0);
      expect(link.sympatheticRiskScore).toBeGreaterThanOrEqual(0.0);
      expect(link.sympatheticRiskScore).toBeLessThanOrEqual(100.0);
    }
  });

  it('identifies critical sympathetic hazard arcade links', () => {
    const highRiskLinks = DEFAULT_ARCADE_LINKS.filter(
      (l) => l.sympatheticRiskScore > 80.0
    );
    expect(highRiskLinks).toHaveLength(1);
    expect(highRiskLinks[0].sourceAr).toBe('AR13664');
  });
});
