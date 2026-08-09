'use client';
import React, { useEffect, useRef, useState } from 'react';

export function WebGPUExperiment() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<string>('Initializing...');
  const [limits, setLimits] = useState<string>('');
  
  useEffect(() => {
    let animationFrameId: number;
    let device: any = null;
    
    async function initWebGPU() {
      if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
        setStatus('WebGPU is not supported by this browser. Falling back to static image.');
        return;
      }
      
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (!adapter) {
          setStatus('Failed to get GPU adapter. Falling back to static image.');
          return;
        }
        
        device = await adapter.requestDevice();
        
        // Inspect and format some limits
        const maxBuffer = device.limits.maxStorageBufferBindingSize;
        const maxComputeInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
        setLimits(`Max Storage Buffer: ${maxBuffer} bytes | Max Compute Workgroup: ${maxComputeInvocations}`);
        
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const context = canvas.getContext('webgpu') as any;
        if (!context) {
          setStatus('Failed to get WebGPU context. Falling back to static image.');
          return;
        }
        
        const presentationFormat = (navigator as any).gpu.getPreferredCanvasFormat();
        context.configure({
          device,
          format: presentationFormat,
          alphaMode: 'premultiplied',
        });
        
        setStatus('Warming up shaders (Idle time compilation)...');
        
        // Wait for idle time to compile shader
        await new Promise<void>(resolve => {
          if ('requestIdleCallback' in window) {
            (window as any).requestIdleCallback(() => resolve());
          } else {
            setTimeout(resolve, 50);
          }
        });
        
        const shaderModule = device.createShaderModule({
          label: 'Triangle Shaders',
          code: `
            @vertex
            fn vertexMain(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
              var pos = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.5),
                vec2<f32>(-0.5, -0.5),
                vec2<f32>(0.5, -0.5)
              );
              return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
            }

            @fragment
            fn fragmentMain() -> @location(0) vec4<f32> {
              return vec4<f32>(0.2, 0.6, 1.0, 1.0);
            }
          `
        });
        
        const pipeline = device.createRenderPipeline({
          label: 'Triangle Pipeline',
          layout: 'auto',
          vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
          },
          fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format: presentationFormat }],
          },
          primitive: {
            topology: 'triangle-list',
          },
        });
        
        setStatus('Rendering active.');
        
        const frame = () => {
          if (!device || !context) return;
          
          const commandEncoder = device.createCommandEncoder();
          const textureView = context.getCurrentTexture().createView();
          
          const renderPassDescriptor: any = {
            colorAttachments: [
              {
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
              },
            ],
          };
          
          const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
          passEncoder.setPipeline(pipeline);
          passEncoder.draw(3);
          passEncoder.end();
          
          device.queue.submit([commandEncoder.finish()]);
          animationFrameId = requestAnimationFrame(frame);
        }
        
        frame();
      } catch (err: any) {
        setStatus(`Error: ${err.message}`);
      }
    }
    
    initWebGPU();
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      if (device) {
        device.destroy();
      }
    };
  }, []);
  
  return (
    <div className="webgpu-experiment border border-slate-700 rounded-lg p-6 bg-slate-900 text-white shadow-xl max-w-2xl">
      <h3 className="text-xl font-bold mb-2">WebGPU Renderer Experiment</h3>
      <div className="flex flex-col gap-2 mb-4 text-sm text-slate-300">
        <p><strong>Status:</strong> {status}</p>
        {limits && <p><strong>Device Limits:</strong> {limits}</p>}
      </div>
      
      <div className="relative w-full aspect-video rounded overflow-hidden bg-black border border-slate-800">
        {!(typeof navigator !== 'undefined' && (navigator as any).gpu) && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-800">
            {/* Static fallback */}
            <p className="text-slate-400">Static Fallback: Triangle Image</p>
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
    </div>
  );
}
