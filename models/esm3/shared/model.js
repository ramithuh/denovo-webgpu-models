/** Complete ESM3 inference graph. Neural primitives are supplied by CPU or WGSL. */
export async function transformer(b,x,{prefix,L,B=1,D,H,F,layers,scale=1,frames,sequenceId,chainId,geometry=0,geomHeads=0,plain=true,zeroFrameless=false,signal,save=()=>{},keep=[]}) {
 const R=L*B;
 for(let i=0;i<layers;i++){
  signal?.throwIfAborted();b.check();const p=`${prefix}.blocks.${i}`;
  if(plain){
   const qkv=await b.linear(await b.norm(x,p+'.attn.layernorm_qkv.0',R,D),p+'.attn.layernorm_qkv.1',R,D,3*D);
   let q=await b.norm(qkv,p+'.attn.q_ln',R,D,{stride:3*D}),k=await b.norm(qkv,p+'.attn.k_ln',R,D,{stride:3*D,offset:D});
   q=await b.rope(q,L,D,H,B);k=await b.rope(k,L,D,H,B);
   const y=await b.attention(q,k,qkv,L,D,H,sequenceId,B);
   x=await b.add(x,await b.linear(y,p+'.attn.out_proj',R,D,D),R*D,scale);
  }
  if(i<geometry){
   const g=p+'.geom_attn',z=await b.linear(await b.norm(x,g+'.s_norm',R,D),g+'.proj',R,D,geomHeads*15);
   const context=await b.geometric(z,frames,sequenceId,chainId,L,geomHeads,B,g,zeroFrameless);
   const y=await b.linear(context,g+'.out_proj',R,geomHeads*3,D);
   x=await b.add(x,y,R*D,scale);
  }
  const z=await b.linear(await b.norm(x,p+'.ffn.0',R,D),p+'.ffn.1',R,D,F*2);
  x=await b.add(x,await b.linear(await b.swiglu(z,R,F),p+'.ffn.3',R,F,D),R*D,scale);
  save(`${prefix}.blocks.${i}`,x,R*D);b.releaseExcept([x,frames,sequenceId,chainId,...keep]);
 }
 return x;
}
async function head(b,x,p,R,D,N){return b.linear(await b.norm(await b.gelu(await b.linear(x,p+'.0',R,D,D),R*D),p+'.2',R,D),p+'.3',R,D,N);}
export async function encodeStructure(b,input,{signal,save=()=>{},keep=[]}={}) {
 const L=input.L,K=Math.min(16,L),frames=await b.frames(input.structure_coords,L);
 const edges=await b.knn(input.structure_coords,frames,input.sequence_id,input.attention_mask,L,K);
 const f=await b.gather(frames,edges,L*K,13),seq=await b.gather(input.sequence_id,edges,L*K,1),chain=b.zeros(L*K);
 let x=await b.relativeEmbedding(edges,input.residue_index,L,K);
 save('encoder.frames',frames,L*13);save('encoder.edges',edges,L*K);save('encoder.embedding',x,L*K*1024);
 x=await transformer(b,x,{prefix:'encoder.transformer',L:K,B:L,D:1024,H:1,F:4096,layers:2,geometry:2,geomHeads:128,plain:false,frames:f,sequenceId:seq,chainId:chain,signal,save,keep:[...keep,frames,edges]});
 const selected=await b.selectLocal(x,frames,L,K,1024);
 const z=await b.linear(selected,'encoder.pre_vq_proj',L,1024,128);save('encoder.pre_vq',z,L*128);
 const tokens=await b.codebook(z,L);save('encoder.tokens',tokens,L);
 return {tokens,frames};
}
export async function forward(b,input,{signal,save=()=>{}}={}) {
 const L=input.L,keep=Object.values(input).filter(v=>b.isTensor(v));
 let frames,structure=input.structure_tokens;
 if(input.encodeStructure){const e=await encodeStructure(b,input,{signal,save,keep});frames=e.frames;structure=await b.structureSpecials(e.tokens,input.sequence_tokens,L);}
 else frames=await b.frames(input.structure_coords,L);
 let x=await b.embedding({...input,structure_tokens:structure});save('esm.encoder',x,L*1536);
 x=await transformer(b,x,{prefix:'esm.transformer',L,D:1536,H:24,F:4096,layers:48,scale:Math.sqrt(48/36),geometry:1,geomHeads:256,zeroFrameless:true,frames,sequenceId:input.sequence_id,chainId:input.chain_id,signal,save,keep});
 const result={embeddings:x,structure_tokens:structure};
 const norm=await b.norm(x,'esm.transformer.norm',L,1536);save('esm.normalized',norm,L*1536);
 for(const [name,n] of [['sequence',64],['structure',4096],['ss8',11],['sasa',19],['function',2080],['residue',1478]]){
  result[name+'_logits']=await head(b,norm,`esm.output_heads.${name}_head`,L,1536,n);
  save(name+'_logits',result[name+'_logits'],L*n);
  b.releaseExcept([norm,...Object.values(result),...keep]);
 }
 return result;
}
export async function decodeStructure(b,tokens,L,{signal,save=()=>{}}={}) {
 const seq=b.zeros(L);
 let x=await b.embeddingLookup(tokens,'decoder.embed.weight',L,1280);save('decoder.embedding',x,L*1280);
 x=await transformer(b,x,{prefix:'decoder.decoder_stack',L,D:1280,H:20,F:3584,layers:30,sequenceId:seq,signal,save,keep:[tokens]});
 x=await b.norm(x,'decoder.decoder_stack.norm',L,1280);save('decoder.normalized',x,L*1280);
 const p='decoder.affine_output_projection';
 const hidden=await b.norm(await b.gelu(await b.linear(x,p+'.ffn1',L,1280,1280),L*1280),p+'.norm',L,1280);
 const projection=await b.linear(hidden,p+'.proj',L,1280,23);save('decoder.projection',projection,L*23);
 const coordinates=await b.backbone(projection,L);save('coordinates',coordinates,L*9);
 const affine=await b.affine(projection,L);save('affine',affine,L*12);
 const h='decoder.plddt_head';
 const plogits=await b.linear(await b.norm(await b.gelu(await b.linear(x,h+'.dense',L,1280,1280),L*1280),h+'.norm',L,1280),h+'.output',L,1280,50);
 const plddt=await b.expectation(plogits,L,50,.01,.02);save('plddt',plddt,L);
 const pair='decoder.pairwise_classification_head';
 const down=await b.linear(x,pair+'.downproject',L,1280,128);
 const pairInput=await b.pairFeatures(down,L);
 const pairLogits=await b.linear(await b.norm(await b.gelu(await b.linear(pairInput,pair+'.linear1',L*L,128,128),L*L*128),pair+'.norm',L*L,128),pair+'.linear2',L*L,128,224);
 save('decoder.pairwise_logits',pairLogits,L*L*224);
 const {pae,ptm}=await b.decoderConfidence(pairLogits,tokens,L);save('predicted_aligned_error',pae,L*L);save('ptm',ptm,1);
 return {coordinates,affine,plddt,pairwise_logits:pairLogits,predicted_aligned_error:pae,ptm};
}
export const OUTPUT_DIMS={embeddings:1536,structure_tokens:1,sequence_logits:64,structure_logits:4096,ss8_logits:11,sasa_logits:19,function_logits:2080,residue_logits:1478};
