import { createPackedLinearRuntime } from '../webgpu/linear.js';
import { iterateWeights } from '../shared/weights.js';
import { metrics } from '../shared/metrics.js';
import { loadGPUWeights } from '../webgpu/weights.js';
async function run() {
  const oracle = await (await fetch('/fixtures/oracle/manifest.json')).json();
  const all = [];
  for (const precision of ['fp32', 'fp16', 'int8']) {
    const manifest = await (await fetch('/weights/' + precision + '/manifest.json')).json();
    const runtime = await createPackedLinearRuntime();
    const startResident = runtime.resources.memoryUsage();
    try {
      for (const filename of oracle.linear_cases) {
        const f = await (await fetch('/fixtures/oracle/' + filename)).json();
        const name = f.name + '.weight';
        let weight;
        for await (const tensor of iterateWeights(manifest, new URL('/weights/' + precision + '/', location.href), { cache: 'none', names: [name] }))
          weight = runtime.uploadWeight(tensor);
        const packedBytes = runtime.resources.memoryUsage() - startResident;
        const input = new Float32Array(f.input.data), bias = f.bias ? new Float32Array(f.bias.data) : undefined;
        const output = await runtime.linear(input, weight, bias);
        const samples = [];
        for (let i = 0; i < 5; i++) {
          const start = performance.now(); await runtime.linear(input, weight, bias); samples.push(performance.now() - start);
        }
        const stats = metrics(output, f.output.data);
        const kernelParity = metrics(output, f.output_by_precision[precision].data);
        if (kernelParity.maxAbsDiff > 1e-4) throw new Error('Linear parity failed: ' + JSON.stringify({ name, precision, ...stats, kernelParity }));
        samples.sort((a,b) => a-b);
        all.push({ precision, layer: f.name, inputShape: f.input.shape, outputShape: f.output.shape, ...stats, kernelParity,
          medianDispatchRoundtripMs: samples[2], packedWeightBytes: packedBytes, peakTrackedBytes: runtime.resources.peakMemoryUsage() });
        runtime.releaseWeight(weight);
        if (runtime.resources.memoryUsage() !== startResident) throw new Error('Linear runtime leaked buffers');
      }
    } finally { runtime.destroy(); }
  }
  const residency=[];
  for(const precision of ['fp32','fp16','int8']){
    const baseURL=new URL('/weights/'+precision+'/',location.href),manifest=await(await fetch(new URL('manifest.json',baseURL))).json();
    const start=performance.now(),store=await loadGPUWeights({manifest,baseURL,cache:'none'});
    const expected=Object.values(manifest.tensors).reduce((sum,t)=>sum+4*Math.ceil(t.shape.reduce((a,b)=>a*b,1)*({F32:4,F16:2,I8:1}[t.dtype])/4)+(t.scale_key?t.shape[0]*4:0),0);
    if(store.byteLength!==expected)throw new Error('Unexpected model weight allocation');
    residency.push({precision,packedWeightBytes:store.byteLength,tensors:store.tensors.size,loadMs:performance.now()-start,
      maxBufferSize:store.resources.device.limits.maxBufferSize,maxStorageBufferBindingSize:store.resources.device.limits.maxStorageBufferBindingSize});
    store.destroy();if(store.resources.memoryUsage()!==0)throw new Error('Model weight store leaked buffers');
  }
  return { linear: all, weightResidency:residency };
}
run().then(postMessage, error => postMessage({ error: error.stack }));
