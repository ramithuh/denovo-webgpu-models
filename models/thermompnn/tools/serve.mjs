import { serve } from "./server.mjs";
const { url } = await serve();
console.log(`ThermoMPNN demo: ${url}/demo/`);
