import { linear, rmsNorm, transition } from './ops.js';
import { decodeWeight } from '../shared/weights.js';
const f32 = Math.fround;
const sigmoid = x => f32(1 / (1 + Math.exp(-x)));
const map = (a, fn) => Float32Array.from(a, fn);
const add = (a, b) => {
  if (a.length !== b.length) throw new Error('Add shape mismatch');
  return Float32Array.from(a, (v, i) => v + b[i]);
};
const mul = (a, b) => Float32Array.from(a, (v, i) => v * b[i]);
function concatRows(arrays, widths, rows) {
  const width = widths.reduce((a,b) => a+b, 0), output = new Float32Array(rows * width);
  for (let i=0;i<rows;i++) { let offset=i*width; arrays.forEach((a,j) => { output.set(a.subarray(i*widths[j],(i+1)*widths[j]),offset); offset+=widths[j]; }); }
  return output;
}
function attention(q, k, v, { queries, keys, channels, heads, bias, allowed, gate, qPreScale=false }) {
  const dim=channels/heads, out=new Float32Array(queries*channels), scores=new Float32Array(keys);
  const scale=f32(1/Math.sqrt(dim));
  for(let i=0;i<queries;i++) for(let h=0;h<heads;h++) {
    let max=-Infinity;
    for(let j=0;j<keys;j++) {
      let sum=0;
      if(allowed && !allowed(i,j)) { scores[j]=-Infinity; continue; }
      for(let c=0;c<dim;c++) {
        const query=q[i*channels+h*dim+c];
        sum=f32(sum+f32((qPreScale?f32(query*scale):query)*k[j*channels+h*dim+c]));
      }
      scores[j]=f32((qPreScale?sum:f32(sum*scale))+(bias?.[(i*keys+j)*heads+h]??0));
      max=Math.max(max,scores[j]);
    }
    if(max===-Infinity) throw new Error('Attention query has no keys');
    let denom=0;
    for(let j=0;j<keys;j++) { scores[j]=f32(Math.exp(scores[j]-max)); denom=f32(denom+scores[j]); }
    for(let j=0;j<keys;j++) scores[j]=f32(scores[j]/denom);
    for(let c=0;c<dim;c++) {
      let sum=0;
      for(let j=0;j<keys;j++) sum=f32(sum+f32(scores[j]*v[j*channels+h*dim+c]));
      const index=i*channels+h*dim+c; out[index]=sum*(gate?.[index]??1);
    }
  }
  return out;
}

export class RFD3CPUModel {
  constructor(weights, config) {
    this.weights=weights; this.config=config;
    for(const weight of weights.values()) { if(!weight.data) weight.data=decodeWeight(weight); delete weight.bytes; delete weight.scales; }
  }
  weight(name) { const w=this.weights.get(name); if(!w) throw new Error('Missing weight '+name); return w; }
  linear(name,x) { return linear(x,this.weight(name+'.weight'),this.weights.get(name+'.bias')?.data); }
  norm(name,x,channels) { const gain=this.weights.get(name+'.weight')?.data; return rmsNorm(x,channels??gain?.length,gain); }
  normLinear(name,x) { return this.linear(name+'.1',this.norm(name+'.0',x)); }
  transition(name,x) { return transition(x,{norm:this.weight(name+'.layer_norm_1.weight').data,w1:this.weight(name+'.linear_1.weight'),w2:this.weight(name+'.linear_2.weight'),w3:this.weight(name+'.linear_3.weight')}); }
  embed(name,features,config) {
    let out;
    for(const feature of Object.keys(config)) {
      if(!features[feature]) throw new Error('Missing feature '+feature);
      const next=this.linear(name+'.embedders.'+feature,features[feature]);
      out=out?add(out,next):next;
    }
    return out;
  }
  cross(name,queries,values,queryCount,keyCount,allowed,heads=4) {
    const q0=this.norm(name+'.ln_q',queries),kv=this.norm(name+'.ln_kv',values);
    const q=this.norm(name+'.q_norm',this.linear(name+'.to_q',q0));
    const k=this.norm(name+'.k_norm',this.linear(name+'.to_k',kv));
    const v=this.linear(name+'.to_v',kv),gate=map(this.linear(name+'.to_g.0',q0),sigmoid);
    const output=attention(q,k,v,{queries:queryCount,keys:keyCount,channels:q.length/queryCount,heads,allowed,gate});
    return this.linear(name+'.to_out.0',output);
  }
  downcast(name,atoms,tokens,tok,conditioning) {
    const count=tokens.length/this.weight(name+'.gca.ln_q.weight').data.length;
    let out=add(tokens,this.cross(name+'.gca',tokens,atoms,count,tok.length,(i,j)=>tok[j]===i));
    if(conditioning) out=add(out,this.normLinear(name+'.process_s',conditioning));
    return out;
  }
  pairformer(name,s,z,count) {
    z=add(z,this.transition(name+'.z_transition',z));
    const p=name+'.attention_pair_bias';
    const normalized=this.norm(p+'.ln_1',s),q=this.linear(p+'.to_q',normalized);
    const k=this.norm(p+'.to_k.ln',this.linear(p+'.to_k',normalized));
    const v=this.norm(p+'.to_v.ln',this.linear(p+'.to_v',normalized));
    const bias=this.linear(p+'.to_b',this.norm(p+'.ln_0',z));
    const gate=map(this.linear(p+'.to_g.0',normalized),sigmoid);
    const heads=this.weight(p+'.to_b.weight').shape[0];
    const result=attention(q,k,v,{queries:count,keys:count,channels:q.length/count,heads,bias,gate,qPreScale:true});
    s=add(s,this.linear(p+'.to_a',result));
    s=add(s,this.transition(name+'.s_transition',s));
    return [s,z];
  }
  relative(name,f) {
    const cfg=this.config.token_initializer.relative_position_encoding, r=cfg.r_max,s=cfg.s_max;
    const bins=2*r+3,width=2*bins+2*s+3,n=f.residue_index.length,encoded=new Float32Array(n*n*width);
    const cyclic=f.cyclic_asym_ids?.[0];
    const length=cyclic===undefined?0:new Set(Array.from(f.residue_index).filter((_,i)=>f.asym_id[i]===cyclic)).size;
    const clip=(x,max)=>Math.max(0,Math.min(max,x));
    for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
      const same=f.asym_id[i]===f.asym_id[j],entity=f.entity_id[i]===f.entity_id[j];
      let offset=f.residue_index[i]-f.residue_index[j];
      if(length && f.asym_id[i]===cyclic && f.asym_id[j]===cyclic) { if(2*offset>length)offset-=length;else if(2*offset < -length)offset+=length; }
      let residue=same?clip(offset+r,2*r):2*r+1;
      let token=same&&f.residue_index[i]===f.residue_index[j]?clip(f.token_index[i]-f.token_index[j]+r,2*r):2*r+1;
      if(f.unindexing_pair_mask[i*n+j]) residue=token=bins-1;
      const chain=entity?clip(f.sym_id[i]-f.sym_id[j]+s,2*s):2*s+1,base=(i*n+j)*width;
      encoded[base+residue]=1;encoded[base+bins+token]=1;encoded[base+2*bins]=Number(entity);encoded[base+2*bins+1+chain]=1;
    }
    return this.linear(name+'.linear',encoded);
  }
  position(name,pos,valid,sinusoidal=false) {
    const n=pos.length/3,mask=new Float32Array(n*n),values=new Float32Array(n*n*(sinusoidal?64:1));
    for(let i=0;i<n;i++)for(let j=0;j<n;j++) {
      let square=0;for(let c=0;c<3;c++) { const delta=f32(pos[i*3+c]-pos[j*3+c]);square=f32(square+f32(delta*delta)); }
      const distance=f32(Math.sqrt(square)),idx=i*n+j;
      mask[idx]=Number(valid(i,j));
      if(sinusoidal)for(let c=0;c<32;c++) {const freq=f32(Math.exp(f32(f32(-Math.log(10000)*c)/32))),angle=f32(distance*freq);values[idx*64+c]=Math.sin(angle);values[idx*64+32+c]=Math.cos(angle);}
      else values[idx]=1/f32(1+Math.max(1e-6,f32(distance*distance)));
    }
    const a=this.linear(name+(sinusoidal?'.output_proj':'.process_inverse_dist'),values);
    const b=this.linear(name+'.process_valid_mask',mask),channels=a.length/(n*n);
    return Float32Array.from(a,(v,i)=>f32(f32(v*mask[Math.floor(i/channels)])+f32(b[i]*mask[Math.floor(i/channels)])));
  }
  initialize(f) {
    const p='token_initializer',cfg=this.config.token_initializer,tok=f.atom_to_token_map;
    const l=tok.length,n=f.residue_index.length,cs=this.config.c_s,cz=this.config.c_z,ca=this.config.c_atom,cp=this.config.c_atompair;
    let s=this.embed(p+'.token_1d_embedder',f,cfg.token_1d_features);
    s=add(s,this.transition(p+'.transition_post_token',s));
    s=this.downcast(p+'.downcast_atom',this.embed(p+'.atom_1d_embedder_1',f,cfg.atom_1d_features),s,tok);
    s=add(s,this.transition(p+'.transition_post_atom',s));s=this.normLinear(p+'.process_s_init',s);
    const zi=this.linear(p+'.to_z_init_i',s),zj=this.linear(p+'.to_z_init_j',s);
    let z=new Float32Array(n*n*cz);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let c=0;c<cz;c++)z[(i*n+j)*cz+c]=zi[j*cz+c]+zj[i*cz+c];
    z=add(z,this.relative(p+'.relative_position_encoding',f));
    z=add(z,this.linear(p+'.process_token_bonds',f.token_bonds));
    const rep=Array.from(f.is_ca).flatMap((v,i)=>v?[i]:[]),pos=new Float32Array(n*3);
    if(rep.length!==n)throw new Error('Expected one representative atom per token');
    rep.forEach((atom,i)=>pos.set(f.ref_pos.subarray(atom*3,atom*3+3),i*3));
    z=add(z,this.position(p+'.ref_pos_embedder_tok',pos,(i,j)=>f.ref_space_uid[rep[i]]===f.ref_space_uid[rep[j]]));
    for(let b=0;b<cfg.n_pairformer_blocks;b++)[s,z]=this.pairformer(p+'.transformer_stack.'+b,s,z,n);
    z=this.normLinear(p+'.process_z_init',concatRows([z,this.relative(p+'.relative_position_encoding2',f)],[cz,cz],n*n));
    for(let b=0;b<2;b++)z=add(z,this.transition(p+'.transition_1.'+b,z));
    const q=this.embed(p+'.atom_1d_embedder_2',f,cfg.atom_1d_features),st=this.normLinear(p+'.process_s_trunk',s);
    const c=Float32Array.from(q,(v,i)=>v+st[tok[Math.floor(i/ca)]*ca+i%ca]);
    let pair=this.position(p+'.motif_pos_embedder',f.motif_pos,(i,j)=>f.is_motif_atom_with_fixed_coord[i]&&f.is_motif_atom_with_fixed_coord[j],true);
    pair=add(pair,this.position(p+'.ref_pos_embedder',f.ref_pos,(i,j)=>f.ref_space_uid[i]===f.ref_space_uid[j]&&f.is_motif_atom_with_fixed_seq[i]&&f.is_motif_atom_with_fixed_seq[j]));
    const left=this.linear(p+'.process_single_l.1',map(c,v=>Math.max(0,v))),right=this.linear(p+'.process_single_m.1',map(c,v=>Math.max(0,v)));
    for(let i=0;i<l;i++)for(let j=0;j<l;j++)for(let k=0;k<cp;k++)pair[(i*l+j)*cp+k]=f32(pair[(i*l+j)*cp+k]+f32(left[i*cp+k]+right[j*cp+k]));
    const zp=this.normLinear(p+'.process_z',z);
    for(let i=0;i<l;i++)for(let j=0;j<l;j++)for(let k=0;k<cp;k++)pair[(i*l+j)*cp+k]=f32(pair[(i*l+j)*cp+k]+zp[(tok[i]*n+tok[j])*cp+k]);
    let mlp=pair;for(const k of [1,3,5])mlp=this.linear(p+'.pair_mlp.'+k,map(mlp,v=>Math.max(0,v)));
    pair=add(pair,mlp);
    const processed=this.linear(p+'.process_pll',pair),pooled=new Float32Array(n*n*cp),temp=new Float32Array(n*l*cp),counts=new Float32Array(n);
    for(let i=0;i<l;i++){counts[tok[i]]++;for(let j=0;j<l;j++)for(let k=0;k<cp;k++){const index=(tok[i]*l+j)*cp+k;temp[index]=f32(temp[index]+processed[(i*l+j)*cp+k]);}}
    for(let i=0;i<n;i++)for(let j=0;j<l;j++)for(let k=0;k<cp;k++){const index=(i*n+tok[j])*cp+k;pooled[index]=f32(pooled[index]+temp[(i*l+j)*cp+k]);}
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let k=0;k<cp;k++)pooled[(i*n+j)*cp+k]/=counts[i]*counts[j];
    z=add(z,this.linear(p+'.project_pll',pooled));
    if(cfg.atom_transformer.n_blocks!==0)throw new Error('Initializer atom transformer is not implemented');
    return { Q_L_init:{data:q,shape:[l,ca]}, C_L:{data:c,shape:[l,ca]}, P_LL:{data:pair,shape:[l,l,cp]}, S_I:{data:s,shape:[n,cs]}, Z_II:{data:z,shape:[n,n,cz]} };
  }
}
