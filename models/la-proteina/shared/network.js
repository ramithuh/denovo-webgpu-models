// Handwritten module hierarchy, transcribed from pinned LD1 and AE1.
// Shapes come from runtime tensors; no recorded operation graphs or step traces.
import { abort } from './loader.js';
import constants from './constants.js';

export class LaProteinaNetwork {
  constructor(ops, weights, config, model) {
    this.o=ops; this.w=weights; this.c=model==='LD1'?config:config.decoder;
    this.model=model;
  }
  linear(x,p) {
    if (!this.w[p+'.weight']) throw Error(`Missing parameter ${p}.weight`);
    return this.o.linear(x,this.w[p+'.weight'],this.w[p+'.bias']);
  }
  norm(x,p) { return this.o.norm(x,this.w[p+'.weight'],this.w[p+'.bias']); }
  transition(x,p) {
    const o=this.o, h=this.linear(x,p+'.swish_linear.0'), n=h.shape.at(-1)/2;
    const a=o.slice(h,h.shape.length-1,0,n), gate=o.slice(h,h.shape.length-1,n,n*2);
    return this.linear(o.mul(a,o.silu(gate)),p+'.linear_out');
  }
  adaptive(x,c,p) {
    const o=this.o, normed=o.norm(x), cond=this.norm(c,p+'.norm_cond');
    return o.add(o.mul(normed,o.sigmoid(this.linear(cond,p+'.to_gamma.0'))),this.linear(cond,p+'.to_beta'));
  }
  scale(x,c,p) { return this.o.mul(x,this.o.sigmoid(this.linear(c,p+'.to_adaln_zero_gamma.0'))); }
  time(tCa,tZ) {
    const o=this.o, D=this.c.t_emb_dim;
    const times=o.tensor([2],[tCa,tZ]);
    return o.kernel([1,2*D],[times],
      `const d=i%${D},k=d%${D/2},f=Math.fround,e=rfExp(f(k*f(-Math.log(2000)/${D/2-1}))),v=f(f(a0[Math.floor(i/${D})]*2000)*e);o[i]=d<${D/2}?Math.sin(v):Math.cos(v);`,
      `let d=i%${D}u;let k=d%${D/2}u;let e=rfExp(f32(k)*${Math.fround(-Math.log(2000)/(D/2-1))});let v=(a0[i/${D}u]*2000.0)*e;let first=(a0[i/${D}u]*2000.0)*rfExp(f32(k-k%4u)*${Math.fround(-Math.log(2000)/(D/2-1))});let sc=rfSinCos(v,first>=125.0);o[i]=select(sc.y,sc.x,d<${D/2}u);`);
  }
  distances(ca) {
    const o=this.o,L=ca.shape[0];
    return o.kernel([L,L],[ca,o.tensor([29],constants.distance_bins)],
      `const r=Math.floor(i/${L})*3,c=i%${L}*3,f=Math.fround;let s=0;for(let k=0;k<3;k++){const d=f(a0[r+k]-a0[c+k]);s=f(s+f(d*d));}const d=f(Math.sqrt(s));let bin=0;for(let k=0;k<29;k++)if(d>a1[k])bin++;o[i]=bin;`,
      `let r=(i/${L}u)*3u;let c=(i%${L}u)*3u;var s=0.0;for(var k=0u;k<3u;k++){let d=a0[r+k]-a0[c+k];s=s+d*d;}let d=sqrt(s);var bin=0u;for(var k=0u;k<29u;k++){if(d>a1[k]){bin++;}}o[i]=f32(bin);`);
  }
  pairFeatures(ca,scCa) {
    const o=this.o,L=ca.shape[0],ld=this.model==='LD1',W=ld?217:157;
    const dist=this.distances(ca),inputs=[dist];
    if(scCa)inputs.push(this.distances(scCa));
    return o.kernel([L,L,W],inputs,
      `const pair=Math.floor(i/${W}),k=i%${W},r=Math.floor(pair/${L}),c=pair%${L},sep=Math.max(0,Math.min(126,r-c+63));o[i]=k<127?Number(k===sep):k<157?Number(k-127===a0[pair]):${scCa?'k<187?Number(k-157===a1[pair]):0':'0'};`,
      `let pair=i/${W}u;let k=i%${W}u;let r=pair/${L}u;let c=pair%${L}u;let sep=u32(clamp(i32(r)-i32(c)+63,0,126));var v=0.0;if(k<127u){v=select(0.0,1.0,k==sep);}else if(k<157u){v=select(0.0,1.0,k-127u==u32(a0[pair]));}${scCa?'else if(k<187u){v=select(0.0,1.0,k-157u==u32(a1[pair]));}':''}o[i]=v;`);
  }
  attention(x,pair,c,p) {
    const o=this.o,L=x.shape[0],H=this.c.nheads,D=this.c.token_dim/H;
    x=this.adaptive(x,c,p+'.adaln');
    const prefix=p+'.mha',node=this.norm(x,prefix+'.node_norm');
    const qkv=this.linear(node,prefix+'.to_qkv'),C=this.c.token_dim;
    const q=o.reshape(this.norm(o.slice(qkv,1,0,C),prefix+'.q_layer_norm'),[L,H,D]);
    const k=o.reshape(this.norm(o.slice(qkv,1,C,2*C),prefix+'.k_layer_norm'),[L,H,D]);
    const v=o.reshape(o.slice(qkv,1,2*C,3*C),[L,H,D]);
    const bias=this.linear(this.norm(pair,prefix+'.pair_norm'),prefix+'.to_bias');
    const sim=o.add(o.mul(o.einsum('ihd,jhd->hij',q,k),1/Math.sqrt(D)),o.permute(bias,[2,0,1]));
    const probs=o.softmax(sim);
    let update=o.reshape(o.einsum('hij,jhd->ihd',probs,v),[L,C]);
    update=o.mul(update,o.sigmoid(this.linear(node,prefix+'.to_g')));
    return this.scale(this.linear(update,prefix+'.to_out_node'),c,p+'.scale_output');
  }
  async forward(input,{signal,onProgress,trace}={}) {
    const o=this.o,{ca,z,scCa,scZ}=input,L=ca.shape[0],ld=this.model==='LD1';
    const keep=[ca,z,scCa,scZ].filter(Boolean);
    const take=async(name,t)=>{if(trace)trace[name]=Array.from(await o.read(t));};
    abort(signal);
    let c,seq,pair;
    if(ld){
      const t=this.time(input.tCa,input.tZ);
      c=this.linear(t,'cond_factory.linear_out');
      c=this.transition(this.transition(c,'transition_c_1'),'transition_c_2');
      const features=o.cat([ca,z,scCa??o.constant([L,3]),scZ??o.constant([L,8]),o.constant([L,24])]);
      seq=this.linear(features,'init_repr_factory.linear_out');
      pair=this.norm(this.linear(this.pairFeatures(ca,scCa),'pair_repr_builder.init_repr_factory.linear_out'),'pair_repr_builder.init_repr_factory.ln_out');
      const pc=this.norm(this.linear(t,'pair_repr_builder.cond_factory.linear_out'),'pair_repr_builder.cond_factory.ln_out');
      pair=this.adaptive(pair,pc,'pair_repr_builder.adaln');
    }else{
      c=o.constant([1,this.c.dim_cond]);
      c=this.transition(this.transition(c,'transition_c_1'),'transition_c_2');
      seq=this.linear(o.cat([ca,z]),'init_repr_factory.linear_out');
      pair=this.linear(this.pairFeatures(ca),'pair_rep_factory.linear_out');
    }
    await take('init',seq);await take('cond',c);await take('pair',pair);
    for(let i=0;i<this.c.nlayers;i++){
      abort(signal);
      const p=`transformer_layers.${i}`;
      seq=o.add(seq,this.attention(seq,pair,c,p+'.mhba'));
      const tr=this.transition(this.adaptive(seq,c,p+'.transition.adaln'),p+'.transition.transition');
      seq=o.add(seq,this.scale(tr,c,p+'.transition.scale_output'));
      await take(`layer.${i}`,seq);
      onProgress?.({phase:ld?'ld1':'ae1',layer:i+1,total:this.c.nlayers});
      await o.checkpoint([...keep,c,pair,seq]);
    }
    if(ld){
      return {ca:this.linear(this.norm(seq,'ca_linear.0'),'ca_linear.1'),z:this.linear(this.norm(seq,'local_latents_linear.0'),'local_latents_linear.1')};
    }
    const logits=this.linear(this.norm(seq,'logit_linear.0'),'logit_linear.1');
    let coords=o.reshape(this.linear(this.norm(seq,'struct_linear.0'),'struct_linear.1'),[L,37,3]);
    coords=o.kernel([L,37,3],[coords,ca],
      `const a=Math.floor(i/3)%37,base=Math.floor(i/111)*3+i%3;o[i]=a===1?a1[base]:${this.c.abs_coors?'a0[i]':'Math.fround(a0[i]+a1[base])'};`,
      `let a=(i/3u)%37u;let base=(i/111u)*3u+i%3u;o[i]=select(${this.c.abs_coors?'a0[i]':'a0[i]+a1[base]'},a1[base],a==1u);`);
    return {coords,logits};
  }
}
