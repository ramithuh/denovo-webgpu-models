import { CPUAlgebra } from './algebra.js';
import { makeEngine } from '../shared/engine.js';
export function createLaProteinaCPU(options={}) {return makeEngine(new CPUAlgebra(),options);}
