import { initialAppState, type AppState } from '../state/appState';
import type { AppAction } from '../actions/appActions';
import { SET_ACTIVE_MEDIA, SET_ACTIVE_SIMULATION, SET_EXPERIENCE_QUALITY, SET_THEME } from '../actions/actionTypes';

export function appReducer(state: AppState = initialAppState, action: AppAction): AppState {
  switch (action.type) {
    case SET_THEME: return { ...state, theme: action.payload };
    case SET_EXPERIENCE_QUALITY: return { ...state, experienceQuality: action.payload };
    case SET_ACTIVE_SIMULATION: return { ...state, activeSimulation: action.payload };
    case SET_ACTIVE_MEDIA: return { ...state, activeMediaId: action.payload };
    default: return state;
  }
}
