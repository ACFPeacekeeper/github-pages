'use client';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Common animation variants and wrappers using Framer Motion.
 */
export const fadeVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
  exit: { opacity: 0 }
};

export const slideUpVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

export { motion, AnimatePresence };
