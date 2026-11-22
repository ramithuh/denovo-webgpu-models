import { CPUBackend } from "./backend.js";
import { loadEngine } from "../shared/engine.js";
export const createBoltzGenCPUEngine = (options) =>
  loadEngine(new CPUBackend(), options);
