import {runBrowser} from './browser.mjs';
const result=await runBrowser(`/test/gpu-harness.js?precision=${process.argv[2]??'fp32'}${process.argv[3]?'&cases='+process.argv[3]:''}`);console.log(JSON.stringify(result,null,2));if(!result.ok)process.exitCode=1;
