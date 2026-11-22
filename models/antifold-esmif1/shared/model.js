/** One execution graph, with CPU reference operations or pure WGSL operations. */
export async function encode(ops,input,trace={},saveTraces=false,signal){
 const save=(name,x)=>{if(saveTraces)trace[name]=x;return x;},check=async keep=>{signal?.throwIfAborted();ops.keep([...keep,...Object.values(trace)]);await ops.checkpoint();};
 const f=await ops.features(input),p='encoder.gvp_encoder',g=p+'.embed_graph';
 if(saveTraces)Object.assign(trace,f);
 const gvp=async([s,v],p,activate=false,gate=false)=>{
  const vhT=await ops.linear(await ops.transposeVectors(v),p+'.wh'),vh=await ops.transposeVectors(vhT,true);
  let outS=await ops.linear(await ops.concat([s,await ops.vectorLength(vh)]),p+'.ws');
  if(activate)outS=await ops.relu(outS);
  let outV=await ops.transposeVectors(await ops.linear(vhT,p+'.wv'),true);
  if(activate)outV=await ops.vectorGate(outV,gate?await ops.linear(outS,p+'.wg'):await ops.vectorLength(outV));
  return [outS,outV];
 };
 const norm=async([s,v],p)=>[await ops.norm(s,p+'.scalar_norm'),await ops.vectorNorm(v)];
 let node=await norm(await gvp([f.nodeScalar,f.nodeVector],g+'.embed_node.0'),g+'.embed_node.1');
 node[0]=await ops.add(node[0],await ops.linear(f.confidenceRBF,g+'.embed_confidence'));
 const edge=await norm(await gvp([f.edgeScalar,f.edgeVector],g+'.embed_edge.0'),g+'.embed_edge.1');
 save('graph.nodeScalar',node[0]);save('graph.nodeVector',node[1]);save('graph.edgeScalar',edge[0]);save('graph.edgeVector',edge[1]);
 const persist=[...Object.values(f),...edge];
 for(let i=0;i<4;i++){
  const b=`${p}.encoder_layers.${i}`;
  let msg=[await ops.concat([await ops.gather(node[0],f.edgeIndex,0),edge[0],await ops.gather(node[0],f.edgeIndex,1)]),await ops.concat([await ops.gather(node[1],f.edgeIndex,0),edge[1],await ops.gather(node[1],f.edgeIndex,1)])];
  for(let j=0;j<3;j++){msg=await gvp(msg,`${b}.conv.message_func.${j}`,j<2,j<2);await check([...persist,...node,...msg]);}
  node=await norm([await ops.add(node[0],await ops.aggregate(msg[0],f.edgeIndex,input.N)),await ops.add(node[1],await ops.aggregate(msg[1],f.edgeIndex,input.N))],b+'.norm.0');
  let ff=await gvp(node,b+'.ff_func.0',true,true);ff=await gvp(ff,b+'.ff_func.1');
  node=await norm([await ops.add(node[0],ff[0]),await ops.add(node[1],ff[1])],b+'.norm.1');
  save(`gvp.${i}.scalar`,node[0]);save(`gvp.${i}.vector`,node[1]);await check([...persist,...node]);
 }
 const padding=ops.tensor(input.padding,[input.N,1]),maskedTokens=ops.tensor(Int32Array.from(input.padding,x=>x?1:32),[input.N,1]);
 const components={};
 components.tokens=await ops.embed(maskedTokens,'encoder.embed_tokens');
 components.diherals=await ops.norm(await ops.linear(f.dihedrals,'encoder.embed_dihedrals.node_embedding'),'encoder.embed_dihedrals.norm_nodes',true);
 components.gvp_out=await ops.linear(await ops.concat([node[0],await ops.rotate(node[1],f.frames)]),'encoder.embed_gvp_output');
 components.confidence=await ops.linear(f.confidenceRBF,'encoder.embed_confidence');
 components.gvp_input_features=await ops.linear(await ops.concat([await ops.slice(f.nodeScalar,0,6),await ops.rotate(f.nodeVector,f.frames)]),'encoder.embed_gvp_input_features');
 let x=components.tokens;for(const key of Object.keys(components).slice(1))x=await ops.add(x,components[key]);
 for(const [k,v]of Object.entries(components))save('embedding.'+k,v);
 x=await ops.position(x,padding,true);save('encoder.input',x);
 for(let i=0;i<8;i++){
  const p=`encoder.layers.${i}`,n=await ops.norm(x,p+'.self_attn_layer_norm');
  x=await ops.add(x,await attention(ops,n,n,p+'.self_attn',padding));
  x=await ops.add(x,await ops.linear(await ops.relu(await ops.linear(await ops.norm(x,p+'.final_layer_norm'),p+'.fc1')),p+'.fc2'));
  save(`encoder.${i}`,x);await check([x,padding]);
 }
 x=await ops.norm(x,'encoder.layer_norm');save('encoder.output',x);
 return {x,padding};
}
async function attention(ops,qx,kvx,p,padding,causal=false){
 const q=await ops.linear(qx,p+'.q_proj'),k=await ops.linear(kvx,p+'.k_proj'),v=await ops.linear(kvx,p+'.v_proj');
 return ops.linear(await ops.attention(q,k,v,padding,causal),p+'.out_proj');
}
export async function decode(ops,encoded,prev,trace={},saveTraces=false,signal){
 const M=prev.length,save=(name,x)=>{if(saveTraces)trace[name]=x;return x;},pad=ops.tensor(Int32Array.from(prev,t=>t===1?1:0),[M,1]);
 let x=await ops.position(await ops.embed(ops.tensor(prev,[M,1]),'decoder.embed_tokens'),pad);save('decoder.input',x);
 for(let i=0;i<8;i++){
  signal?.throwIfAborted();const p=`decoder.layers.${i}`,n=await ops.norm(x,p+'.self_attn_layer_norm');
  x=await ops.add(x,await attention(ops,n,n,p+'.self_attn',pad,true));
  x=await ops.add(x,await attention(ops,await ops.norm(x,p+'.encoder_attn_layer_norm'),encoded.x,p+'.encoder_attn',encoded.padding));
  x=await ops.add(x,await ops.linear(await ops.relu(await ops.linear(await ops.norm(x,p+'.final_layer_norm'),p+'.fc1')),p+'.fc2'));save(`decoder.${i}`,x);
  ops.keep([x,pad,encoded.x,encoded.padding,...Object.values(trace)]);await ops.checkpoint();
 }
 return save('logits',await ops.linear(await ops.norm(x,'decoder.layer_norm'),'decoder.output_projection'));
}
