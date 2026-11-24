import { RFD3CPUModel } from './model.js';
import { rmsNorm } from './ops.js';
const f32=Math.fround,add=(a,b)=>Float32Array.from(a,(v,i)=>v+b[i]),mul=(a,b)=>Float32Array.from(a,(v,i)=>v*b[i]);
const sigmoid=x=>1/(1+Math.exp(-x));
const map=(a,f)=>Float32Array.from(a,f);
function distogram(coords,rep,sigma=16) {
  const n=rep.length,out=new Float32Array(n*n*65);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
    let sum=0;for(let c=0;c<3;c++){const d=f32(coords[rep[i]*3+c]-coords[rep[j]*3+c]);sum=f32(sum+f32(d*d));}
    const distance=f32(Math.sqrt(sum));let bin=0;
    while(bin<64&&distance>f32(1/sigma+bin*f32((30/sigma-1/sigma)/63)))bin++;
    out[(i*n+j)*65+bin]=1;
  }
  return out;
}
function concatRows(arrays,widths,rows) {
  const width=widths.reduce((a,b)=>a+b,0),out=new Float32Array(rows*width);
  for(let r=0;r<rows;r++){let pos=r*width;for(let a=0;a<arrays.length;a++){out.set(arrays[a].subarray(r*widths[a],(r+1)*widths[a]),pos);pos+=widths[a];}}
  return out;
}
function neighbours(f,coords,tok,count,near) {
  const l=tok.length,k=Math.min(l,count),n=f.asym_id.length,sets=[];
  const multi=new Set(f.asym_id).size>3;
  if(k===l&&!multi)return Array.from({length:l},()=>null);
  const members=Array.from({length:n},()=>[]);tok.forEach((t,i)=>members[t].push(i));
  const full=Array.from({length:n},()=>new Uint8Array(n));
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)full[i][j]=Math.abs(i-j)<=near&&members[i].every(a=>members[j].every(b=>Math.abs(a-b)<=Math.floor(k/2)));
  for(let i=0;i<l;i++) {
    if(multi){
      const chain=Array.from(tok,t=>f.asym_id[t]),base=Array.from(tok,t=>!f.unindexing_pair_mask[tok[i]*n+t]),distance=Float32Array.from({length:l},(_,j)=>Math.hypot(...[0,1,2].map(c=>f32(coords[i*3+c]-coords[j*3+c]))));
      const inter=Array.from({length:l},(_,j)=>j).filter(j=>base[j]&&chain[j]!==chain[i]&&distance[j]<12),ranks=new Map();
      for(const c of new Set(chain)){const cols=inter.filter(j=>chain[j]===c).sort((a,b)=>distance[a]-distance[b]||a-b);cols.forEach((j,r)=>ranks.set(j,r*12+distance[j]));}
      inter.sort((a,b)=>ranks.get(a)-ranks.get(b)||a-b);const reserved=new Set(inter.slice(0,Math.max(32,Math.floor(k/4))));
      const offset=Math.max(...distance)+1;
      const score=j=>!base[j]||reserved.has(j)?Infinity:chain[j]===chain[i]&&Math.abs(tok[i]-tok[j])<=near?-1:f32(distance[j]+(chain[j]===chain[i]?0:offset));
      const rest=Array.from({length:l},(_,j)=>j).sort((a,b)=>score(a)-score(b)||a-b);
      if(l<=32){const size=2**Math.ceil(Math.log2(l)),order=Array.from({length:size},(_,i)=>i),value=i=>i<l?score(i):Infinity;const maximum=Math.max(...order.slice(0,l).map(value));order.splice(0,l,...order.slice(0,l).filter(i=>value(i)<maximum),...order.slice(0,l).filter(i=>value(i)===maximum));
        for(let span=2;span<=size;span*=2)for(let stride=span/2;stride>=1;stride/=2)for(let a=0;a<size;a++){const b=a^stride;if(b>a&&(value(order[a])<value(order[b]))===Boolean(a&span))[order[a],order[b]]=[order[b],order[a]];}rest.splice(0,rest.length,...order.filter(i=>i<l));}
      const intra=k-Math.max(32,Math.floor(k/4)),chosen=[...reserved,...rest.slice(0,intra),...rest.slice(intra+reserved.size,k)],counts=new Map();for(const j of chosen)counts.set(j,(counts.get(j)??0)+1);sets.push(counts);continue;
    }
    const forced=[];
    for(let j=0;j<l;j++)if(full[tok[i]][tok[j]]&&f.asym_id[tok[i]]===f.asym_id[tok[j]]&&!f.unindexing_pair_mask[tok[i]*n+tok[j]])forced.push(j);
    const known=new Set(forced);
    const rest=Array.from({length:l},(_,j)=>j).filter(j=>!known.has(j));
    const distances=new Float32Array(l);
    for(const j of rest){let square=0;for(let c=0;c<3;c++){const delta=f32(coords[i*3+c]-coords[j*3+c]);square=f32(square+f32(delta*delta));}distances[j]=Math.sqrt(square);}
    rest.sort((a,b)=>distances[a]-distances[b]||a-b);
    sets.push(new Set(forced.concat(rest.slice(0,k-forced.length))));
  }
  return sets;
}

export class RFD3CPUDenoiser extends RFD3CPUModel {
  ada(name,a,s,width) {
    const normalized=rmsNorm(a,width),cond=this.norm(name+'.ln_s',s);
    return add(mul(map(this.linear(name+'.to_gain.0',cond),sigmoid),normalized),this.linear(name+'.to_bias',cond));
  }
  block(name,a,s,z,rows,sets) {
    const p=name+'.attention_pair_bias',width=a.length/rows;
    const normalized=this.ada(p+'.ada_ln_1',a,s,width);
    const q=this.norm(p+'.ln_q',this.linear(p+'.to_q',normalized)),k=this.norm(p+'.ln_k',this.linear(p+'.to_k',normalized));
    const v=this.linear(p+'.to_v',normalized),gate=map(this.linear(p+'.to_g.0',normalized),sigmoid),bias=this.linear(p+'.to_b',z);
    const heads=this.weight(p+'.to_b.weight').shape[0],dim=width/heads,out=new Float32Array(a.length),scores=new Float32Array(rows);
    for(let i=0;i<rows;i++)for(let h=0;h<heads;h++) {
      let max=-Infinity;
      for(let j=0;j<rows;j++) {
        if(sets?.[i]&&!sets[i].has(j)){scores[j]=-Infinity;continue;}
        let dot=0;for(let c=0;c<dim;c++)dot=f32(dot+f32(q[i*width+h*dim+c]*k[j*width+h*dim+c]));
        scores[j]=f32(f32(dot/Math.sqrt(dim))+bias[(i*rows+j)*heads+h]);if(sets?.[i] instanceof Map)scores[j]=f32(scores[j]+Math.log(sets[i].get(j)));max=Math.max(max,scores[j]);
      }
      let sum=0;for(let j=0;j<rows;j++){scores[j]=Math.exp(scores[j]-max);sum=f32(sum+scores[j]);}
      for(let j=0;j<rows;j++)scores[j]/=sum;
      for(let c=0;c<dim;c++){let total=0;for(let j=0;j<rows;j++)total=f32(total+f32(scores[j]*v[j*width+h*dim+c]));const idx=i*width+h*dim+c;out[idx]=total*gate[idx];}
    }
    a=add(a,mul(this.linear(p+'.to_o',out),map(this.linear(p+'.linear_output_project.0',s),sigmoid)));
    const tr=name+'.transition_block',at=this.ada(tr+'.ada_ln',a,s,width);
    const x=this.linear(tr+'.linear_1',at),y=this.linear(tr+'.linear_2',at);
    const swiglu=Float32Array.from(x,(v,i)=>f32(v/(1+Math.exp(-v)))*y[i]);
    return add(a,mul(map(this.linear(tr+'.linear_output_project.0',s),sigmoid),this.linear(tr+'.linear_3',swiglu)));
  }
  time(index,times) {
    const name='diffusion_module',w=this.weight(name+'.fourier_embedding.'+index+'.w').data,b=this.weight(name+'.fourier_embedding.'+index+'.b').data;
    const out=new Float32Array(times.length*w.length),pi2=f32(2*f32(Math.PI));
    for(let i=0;i<times.length;i++) {
      const time=f32(f32(Math.log(f32(Math.max(times[i],1e-20)/this.config.diffusion_module.sigma_data)))/4);
      for(let c=0;c<w.length;c++)out[i*w.length+c]=Math.cos(f32(pi2*f32(f32(time*w[c])+b[c])));
    }
    const result=this.normLinear(name+'.process_n.'+index,out),channels=result.length/times.length;
    for(let i=0;i<times.length;i++)if(times[i]===0)result.fill(0,i*channels,(i+1)*channels);
    return result;
  }
  denoise({features:f,X_noisy_L:input,t,n_recycle=this.config.diffusion_module.n_recycle,initializer}) {
    if(!Number.isInteger(n_recycle)||n_recycle<1)throw new Error('n_recycle must be positive');
    const tok=f.atom_to_token_map,l=tok.length,n=f.residue_index.length,p='diffusion_module',cfg=this.config.diffusion_module;
    if(input.length!==l*3||!Number.isFinite(t)||t<0)throw new Error('Expected one coordinate sample and nonnegative finite noise');
    const init=initializer??this.initialize(f),q0=init.Q_L_init.data,c0=init.C_L.data,pair=init.P_LL.data,s0=init.S_I.data,z0=init.Z_II.data;
    const rep=Array.from(f.is_ca).flatMap((v,i)=>v?[i]:[]),sigma=cfg.sigma_data;
    const tl=Float32Array.from(f.is_motif_atom_with_fixed_coord,v=>v?0:t),ti=Float32Array.from(f.is_motif_token_with_fully_fixed_coord,v=>v?0:t);
    const uniform=Float32Array.from(input,v=>v/f32(Math.sqrt(f32(f32(t*t)+sigma*sigma))));
    const scaled=Float32Array.from(input,(v,i)=>v/f32(Math.sqrt(f32(f32(tl[Math.floor(i/3)]**2)+sigma*sigma))));
    const atomKeys=neighbours(f,input,tok,cfg.n_attn_keys,cfg.n_attn_seq_neighbours);
    const projected=this.linear(p+'.process_a.linear',scaled),counts=new Float32Array(n),a0=new Float32Array(n*cfg.c_token);
    for(let i=0;i<l;i++){counts[tok[i]]++;for(let c=0;c<cfg.c_token;c++){const idx=tok[i]*cfg.c_token+c;a0[idx]=f32(a0[idx]+projected[i*cfg.c_token+c]);}}
    for(let i=0;i<n;i++)for(let c=0;c<cfg.c_token;c++)a0[i*cfg.c_token+c]/=counts[i];
    let c=add(c0,this.time(0,tl));c=add(c,this.normLinear(p+'.process_c',c));
    const baseS=add(this.downcast(p+'.downcast_c',c0,s0,tok),this.time(1,ti));
    let q=add(q0,this.linear(p+'.process_r',scaled));
    for(let i=0;i<cfg.atom_attention_encoder.n_blocks;i++)q=this.block(p+'.encoder.blocks.'+i,q,c,pair,l,atomKeys);
    const baseQ=q,baseA=this.downcast(p+'.downcast_q',q,a0,tok,baseS);
    let lastCoords,lastDistance,logits,indices;
    for(let cycle=0;cycle<n_recycle;cycle++) {
      let s=baseS,z=z0,a=baseA;q=baseQ;
      const enc=p+'.diffusion_token_encoder',ecfg=cfg.diffusion_token_encoder;
      for(let i=0;i<2;i++)s=add(s,this.transition(enc+'.transition_1.'+i,s));
      const parts=[z],widths=[this.config.c_z];
      if(ecfg.use_distogram){
        if(ecfg.use_sinusoidal_distogram_embedder)throw new Error('This checkpoint uses bucketed diffusion distances');
        parts.push(distogram(uniform,rep,sigma));widths.push(65);
      }
      if(ecfg.use_self){parts.push(lastDistance??new Float32Array(n*n*65));widths.push(65);}
      z=this.normLinear(enc+'.process_z',concatRows(parts,widths,n*n));
      for(let i=0;i<2;i++)z=add(z,this.transition(enc+'.transition_2.'+i,z));
      for(let i=0;i<ecfg.n_pairformer_blocks;i++)[s,z]=this.pairformer(enc+'.pairformer_stack.'+i,s,z,n);
      const coords=lastCoords??input,tokenPos=new Float32Array(n*3);rep.forEach((v,i)=>tokenPos.set(coords.subarray(v*3,v*3+3),i*3));
      const tokenKeys=neighbours(f,tokenPos,Int32Array.from({length:n},(_,i)=>i),cfg.diffusion_transformer.n_keys,cfg.diffusion_transformer.n_local_tokens).map(keys=>keys instanceof Map?new Set(keys.keys()):keys);
      for(let i=0;i<cfg.diffusion_transformer.n_block;i++)a=this.block(p+'.diffusion_transformer.blocks.'+i,a,s,z,n,tokenKeys);
      const split=cfg.atom_attention_decoder.upcast.n_split;
      for(let i=0;i<cfg.atom_attention_decoder.n_blocks;i++){
        q=add(q,this.cross(p+'.decoder.upcast.'+i+'.gca',q,a,l,n*split,(atom,key)=>tok[atom]===Math.floor(key/split)));
        q=this.block(p+'.decoder.atom_transformer.'+i,q,c,pair,l,atomKeys);
      }
      a=this.downcast(p+'.decoder.downcast',q,a,tok,s);
      const update=this.normLinear(p+'.to_r_update',q);
      lastCoords=Float32Array.from(update,(v,i)=>{
        const time=tl[Math.floor(i/3)],denom=f32(sigma*sigma+f32(time*time));
        return f32(f32(f32(sigma*sigma/denom)*input[i])+f32(f32(sigma*time/f32(Math.sqrt(denom)))*v));
      });
      logits=this.linear(p+'.sequence_head.linear',a);
      const valid=this.config.sequence_valid_out_mask;
      indices=new Int32Array(n);
      for(let i=0;i<n;i++){let maximum=-Infinity;for(let c=0;c<32;c++)maximum=Math.max(maximum,logits[i*32+c]);const prob=Float32Array.from({length:32},(_,c)=>Math.exp(logits[i*32+c]-maximum));let sum=0;for(const v of prob)sum=f32(sum+v);let best=0;for(let c=0;c<32;c++){const value=valid[c]?f32(prob[c]/sum):0;if(value>best){best=value;indices[i]=c;}}}
      lastDistance=distogram(lastCoords,rep,1);
    }
    return {X_L:{data:lastCoords,shape:[1,l,3]},sequence_logits_I:{data:logits,shape:[1,n,32]},sequence_indices_I:{data:indices,shape:[1,n]}};
  }
}
