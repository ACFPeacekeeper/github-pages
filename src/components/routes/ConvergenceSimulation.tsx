'use client';

import { useEffect, useMemo, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { createSimulationController } from '../../simulations/context/createSimulationController';
import { getSimulationScenario, SIMULATION_SCENARIOS } from '../../simulations/repository/scenarios';
import type { SimulationSnapshot } from '../../simulations/state/types';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { useAppDispatch } from '../../redux/store/hooks';
import { setActiveSimulation } from '../../redux/actions/appActions';

export default function ConvergenceSimulation() {
  const [scenarioId, setScenarioId] = useState('balanced');
  const dispatch = useAppDispatch();
  const controller = useMemo(() => createSimulationController(getSimulationScenario(scenarioId)), [scenarioId]);
  const [snapshot, setSnapshot] = useState<SimulationSnapshot>(controller.initial);
  const reducedMotion = useReducedMotion();
  useEffect(() => { dispatch(setActiveSimulation(scenarioId)); }, [dispatch, scenarioId]);

  useEffect(() => setSnapshot(controller.initial), [controller]);
  useEffect(() => {
    if (snapshot.status !== 'running') return;
    const timer = window.setInterval(() => setSnapshot((value) => controller.advance(value)), reducedMotion ? 300 : 120);
    return () => window.clearInterval(timer);
  }, [controller, reducedMotion, snapshot.status]);

  const visible = snapshot.points.slice(0, snapshot.cursor + 1);
  const current = visible[visible.length - 1];
  const chartWidth = 640;
  const chartHeight = 220;
  const max = Math.max(...snapshot.points.map((point) => point.incumbent));
  const min = Math.min(...snapshot.points.map((point) => point.lowerBound));
  const toPath = (key: 'incumbent' | 'lowerBound') => visible.map((point, index) => {
    const x = (point.iteration / (snapshot.points.length - 1)) * chartWidth;
    const y = chartHeight - ((point[key] - min) / (max - min)) * chartHeight;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');

  return (
    <section className="simulation-panel" aria-labelledby="simulation-title">
      <div className="simulation-header">
        <div><p className="eyebrow">Interactive simulation · deterministic demo</p><h2 id="simulation-title">Watch an optimizer converge.</h2><p>Compare the improving incumbent solution with its mathematical lower bound.</p></div>
        <label>Search strategy<select value={scenarioId} onChange={(event) => setScenarioId(event.target.value)}>{SIMULATION_SCENARIOS.map((scenario) => <option key={scenario.id} value={scenario.id}>{scenario.name}</option>)}</select></label>
      </div>
      <div className="simulation-chart">
        <svg viewBox={`-12 -12 ${chartWidth + 24} ${chartHeight + 24}`} role="img" aria-labelledby="simulation-chart-title" aria-describedby="simulation-chart-description">
          <title id="simulation-chart-title">Optimization convergence chart</title>
          {[0, .25, .5, .75, 1].map((ratio) => <line key={ratio} x1="0" x2={chartWidth} y1={chartHeight * ratio} y2={chartHeight * ratio} className="simulation-gridline" />)}
          <path d={toPath('lowerBound')} className="simulation-line simulation-line--bound" />
          <path d={toPath('incumbent')} className="simulation-line simulation-line--incumbent" />
        </svg>
        <p id="simulation-chart-description" className="sr-only">At iteration {current.iteration}, the incumbent cost is {current.incumbent} and lower bound is {current.lowerBound}.</p>
      </div>
      <div className="simulation-footer">
        <div className="simulation-controls">
          <button type="button" onClick={() => setSnapshot((value) => value.status === 'complete' ? { ...controller.reset(), status: 'running' } : { ...value, status: value.status === 'running' ? 'idle' : 'running' })} aria-label={snapshot.status === 'running' ? 'Pause simulation' : 'Run simulation'}>{snapshot.status === 'running' ? <Pause size={16} /> : <Play size={16} />} {snapshot.status === 'running' ? 'Pause' : snapshot.status === 'complete' ? 'Replay' : 'Run'}</button>
          <button type="button" onClick={() => setSnapshot(controller.reset())}><RotateCcw size={16} /> Reset</button>
        </div>
        <dl aria-live="polite"><div><dt>Iteration</dt><dd>{current.iteration + 1}/{snapshot.points.length}</dd></div><div><dt>Incumbent</dt><dd>{current.incumbent}</dd></div><div><dt>Lower bound</dt><dd>{current.lowerBound}</dd></div><div><dt>Gap</dt><dd>{(((current.incumbent - current.lowerBound) / current.incumbent) * 100).toFixed(1)}%</dd></div></dl>
      </div>
    </section>
  );
}
