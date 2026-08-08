import { ExperienceQuality } from '../../enums/ExperienceQuality';

export interface AppState {
  theme: 'light' | 'dark';
  experienceQuality: ExperienceQuality;
  activeSimulation: string | null;
  activeMediaId: string | null;
}

export const initialAppState: AppState = {
  theme: 'dark',
  experienceQuality: ExperienceQuality.Full,
  activeSimulation: null,
  activeMediaId: null,
};
