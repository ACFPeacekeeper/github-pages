import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
import ModelViewer from '../../../../src/frameworks/react/components/models/ModelViewer';

// Mock IntersectionObserver
const mockIntersectionObserver = vi.fn();
mockIntersectionObserver.mockReturnValue({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
});
window.IntersectionObserver = mockIntersectionObserver as any;

// Mock matchMedia
window.matchMedia = vi.fn().mockImplementation((query) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener: vi.fn(), // deprecated
  removeListener: vi.fn(), // deprecated
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
}));

// Mock ResizeObserver
window.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock Three.js since WebGL is not available in jsdom
vi.mock('three', () => {
  return {
    WebGLRenderer: vi.fn().mockImplementation(() => ({
      render: vi.fn(),
      setSize: vi.fn(),
      setPixelRatio: vi.fn(),
      dispose: vi.fn(),
      info: { memory: { geometries: 0, textures: 0 } }
    })),
    Scene: vi.fn().mockImplementation(() => ({
      add: vi.fn(),
      rotation: { x: 0, y: 0, z: 0 },
      traverse: vi.fn(),
    })),
    PerspectiveCamera: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() },
      lookAt: vi.fn(),
      updateProjectionMatrix: vi.fn(),
      matrixWorldInverse: {},
      projectionMatrix: {},
    })),
    Vector3: vi.fn().mockImplementation(() => ({
      set: vi.fn(),
      project: vi.fn().mockImplementation(function(this: any) { this.x = 0; this.y = 0; this.z = 0; return this; }),
      x: 0, y: 0, z: 0
    })),
    AmbientLight: vi.fn(),
    DirectionalLight: vi.fn().mockImplementation(() => ({
      position: { set: vi.fn() }
    })),
  };
});

vi.mock('three/examples/jsm/loaders/GLTFLoader.js', () => {
  return {
    GLTFLoader: vi.fn().mockImplementation(() => ({
      setDRACOLoader: vi.fn(),
      load: vi.fn((url, onLoad) => {
        // Simulate immediate load success
        onLoad({ scene: {} });
      }),
    }))
  };
});

vi.mock('three/examples/jsm/loaders/DRACOLoader.js', () => {
  return {
    DRACOLoader: vi.fn().mockImplementation(() => ({
      setDecoderPath: vi.fn(),
      dispose: vi.fn(),
    }))
  };
});


describe('ModelViewer Component', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('renders a canvas for the model', () => {
    render(<ModelViewer url="/test.glb" alt="Test Model" />);
    
    expect(screen.getByLabelText('Test Model')).toBeInTheDocument();
    expect(screen.getByLabelText('Test Model')).toHaveClass('block');
    
    // Check loading indicator initially
    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  it('simulates visibility to trigger mount and load', async () => {
    // We capture the observer callback to trigger it
    let intersectionCallback: any;
    window.IntersectionObserver = vi.fn().mockImplementation((cb) => {
      intersectionCallback = cb;
      return {
        observe: vi.fn(),
        unobserve: vi.fn(),
        disconnect: vi.fn(),
      };
    }) as any;

    render(
      <ModelViewer 
        url="/test.glb" 
        alt="Test Model" 
        cameraPresets={[{ name: 'Front', position: [0, 0, 5] }]} 
        annotations={[{ id: 'a1', position: [0, 1, 0], label: 'Test Note' }]} 
      />
    );

    // Trigger intersection
    act(() => {
      if (intersectionCallback) {
        intersectionCallback([{ isIntersecting: true }]);
      }
    });

    // Advance timers for GLTFLoader mock to fire onLoad
    await act(async () => {
      vi.runAllTimers();
      // Need to flush promises because dynamic imports are used in mount()
      await new Promise(resolve => process.nextTick(resolve));
    });

    // Loading should be gone
    expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();

    // The preset button should be visible
    expect(screen.getByRole('button', { name: 'Front' })).toBeInTheDocument();
    
    // Annotations should be visible since projection is mocked to valid coordinates
    expect(screen.getByText('Test Note')).toBeInTheDocument();
  });
});
