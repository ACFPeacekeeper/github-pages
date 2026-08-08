import { describe, expect, it } from 'vitest';
import { setActiveMedia, setActiveSimulation, setExperienceQuality, setTheme } from '../../../src/redux/actions/appActions';
import { appReducer } from '../../../src/redux/reducers/appReducer';
import { initialAppState } from '../../../src/redux/state/appState';
import { ExperienceQuality } from '../../../src/enums/ExperienceQuality';

describe('appReducer', () => {
  it('updates theme state', () => expect(appReducer(initialAppState, setTheme('light')).theme).toBe('light'));
  it('updates the selected quality tier', () => expect(appReducer(initialAppState, setExperienceQuality(ExperienceQuality.Reduced)).experienceQuality).toBe(ExperienceQuality.Reduced));
  it('tracks active simulations and media independently', () => {
    const simulationState = appReducer(initialAppState, setActiveSimulation('balanced'));
    const mediaState = appReducer(simulationState, setActiveMedia('media-reel:2'));
    expect(mediaState.activeSimulation).toBe('balanced');
    expect(mediaState.activeMediaId).toBe('media-reel:2');
  });
});
