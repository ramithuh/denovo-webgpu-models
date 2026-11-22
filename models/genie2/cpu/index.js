import { CPUAlgebra } from "./algebra.js";
import { initializeEngine } from "../shared/engine.js";
export async function createGenie2CPUEngine(options) {
  return initializeEngine(new CPUAlgebra(), options, { backend: "cpu" });
}
export { createGenie2CPUEngine as createGenie2Engine };
