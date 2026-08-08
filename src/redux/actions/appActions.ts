import { ExperienceQuality } from '../../enums/ExperienceQuality';
import { SET_ACTIVE_MEDIA, SET_ACTIVE_SIMULATION, SET_EXPERIENCE_QUALITY, SET_THEME } from './actionTypes';

export const setTheme = (theme: 'light' | 'dark') => ({ type: SET_THEME, payload: theme } as const);
export const setExperienceQuality = (quality: ExperienceQuality) => ({ type: SET_EXPERIENCE_QUALITY, payload: quality } as const);
export const setActiveSimulation = (simulationId: string | null) => ({ type: SET_ACTIVE_SIMULATION, payload: simulationId } as const);
export const setActiveMedia = (mediaId: string | null) => ({ type: SET_ACTIVE_MEDIA, payload: mediaId } as const);

export type AppAction = ReturnType<typeof setTheme | typeof setExperienceQuality | typeof setActiveSimulation | typeof setActiveMedia>;
