import { ExperienceQuality } from '../enums/ExperienceQuality';

export const VISUAL_EXPERIENCE = {
  maxDevicePixelRatio: 1.75,
  reducedDevicePixelRatio: 1,
  modelRotationStep: Math.PI / 12,
  defaultQuality: ExperienceQuality.Full,
  constellationViewBox: { width: 720, height: 420 },
} as const;
