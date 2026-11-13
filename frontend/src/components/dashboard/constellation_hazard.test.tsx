import { describe, it, expect } from 'vitest';
import {
  DEFAULT_CONSTELLATION_FLEET,
  type SatelliteHazardEntry,
} from './ConstellationHazardMatrix';

describe('ConstellationHazardMatrix Component Test Suite', () => {
  it('validates fleet configuration for NavIC, GSAT, and LEO assets', () => {
    expect(DEFAULT_CONSTELLATION_FLEET).toHaveLength(4);
    const navic = DEFAULT_CONSTELLATION_FLEET.find((s) => s.constellation === 'NavIC');
    expect(navic).toBeDefined();
    expect(navic?.orbitType).toBe('GSO');
    expect(navic?.altitudeKm).toBe(35786);
  });

  it('verifies safe-mode trigger conditions for high-flux satellites', () => {
    const criticalAssets = DEFAULT_CONSTELLATION_FLEET.filter(
      (s) => s.hazardState === 'CRITICAL_SAFE_MODE'
    );
    expect(criticalAssets).toHaveLength(1);
    expect(criticalAssets[0].satelliteId).toContain('GSAT-7A');
    expect(criticalAssets[0].sepProtonFluxPfu).toBeGreaterThan(500.0);
    expect(criticalAssets[0].seuProbabilityPerHour).toBeGreaterThan(0.20);
  });

  it('validates low-Earth orbit geomagnetic shielding mitigation', () => {
    const leoAsset = DEFAULT_CONSTELLATION_FLEET.find((s) => s.orbitType === 'SSO / LEO');
    expect(leoAsset).toBeDefined();
    expect(leoAsset?.hazardState).toBe('NOMINAL');
    expect(leoAsset?.doseRateRadPerHr).toBeLessThan(0.5);
  });
});
