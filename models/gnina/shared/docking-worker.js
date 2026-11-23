import {searchChain} from "./docking-search.js";
let context;
self.onmessage=async ({data})=>{
  try {
    if(data.context) context=data.context;
    const {tree,grid,options}=context;
    const bank=await searchChain(tree,grid,options,data.id);
    self.postMessage({id:data.id,bank});
  } catch(error){self.postMessage({error:String(error)});}
};
