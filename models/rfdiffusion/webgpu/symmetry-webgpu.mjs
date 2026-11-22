import { describeSymmetry, rotationCoefficients } from "../shared/symmetry.mjs";

const workgroupSize = 64;
const shader = `
@group(0) @binding(0) var<storage, read> unit: array<f32>;
@group(0) @binding(1) var<storage, read> rotations: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(${workgroupSize})
fn expand(@builtin(global_invocation_id) id: vec3<u32>) {
  let dst = 3u * id.x;
  if (dst >= arrayLength(&result)) { return; }
  let src = dst % arrayLength(&unit);
  let rot = 3u * (dst / arrayLength(&unit));
  let c = rotations[rot];
  let s = rotations[rot + 1u];
  let sign = rotations[rot + 2u];
  let x = unit[src];
  let y = unit[src + 1u];
  result[dst] = c * x - s * y;
  result[dst + 1u] = sign * (s * x + c * y);
  result[dst + 2u] = sign * unit[src + 2u];
}`;

/** Compile once for a caller-owned GPUDevice, then rotate fixed coordinate arrays. */
export async function createSymmetryExpander(device) {
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: shader }),
      entryPoint: "expand",
    },
  });
  return async function expand(group, coordinates) {
    const shape = describeSymmetry(group, coordinates);
    const bytes = shape.copies * coordinates.byteLength;
    const workgroups = Math.ceil(bytes / 12 / workgroupSize);
    if (
      bytes >
        Math.min(
          device.limits.maxStorageBufferBindingSize,
          device.limits.maxBufferSize,
        ) ||
      workgroups > device.limits.maxComputeWorkgroupsPerDimension
    ) {
      throw new RangeError(
        "Expanded coordinates exceed the device buffer or dispatch limit",
      );
    }
    const buffers = [];
    function allocate(size, usage) {
      const buffer = device.createBuffer({ size, usage });
      buffers.push(buffer);
      return buffer;
    }
    try {
      const rotations = rotationCoefficients(shape);
      const inputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const unit = allocate(coordinates.byteLength, inputUsage);
      const coefficients = allocate(rotations.byteLength, inputUsage);
      const output = allocate(
        bytes,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const readback = allocate(
        bytes,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      );
      device.queue.writeBuffer(unit, 0, coordinates);
      device.queue.writeBuffer(coefficients, 0, rotations);
      const group0 = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [unit, coefficients, output].map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      const commands = device.createCommandEncoder();
      const pass = commands.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group0);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      commands.copyBufferToBuffer(output, 0, readback, 0, bytes);
      device.queue.submit([commands.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return new Float32Array(readback.getMappedRange().slice(0));
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
