import { CPUAlgebra } from "./algebra.js";
import { makeEngine } from "../shared/engine.js";
export const createRFdiffusionEngine = (options) =>
  makeEngine(new CPUAlgebra(), options);
