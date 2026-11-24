// Dynamic model orchestration. Tensor math is implemented by WGSL GPUOps;
// JavaScript handles dimensions, feature metadata and resource ownership only.
export class RFD3GPUModel {
  constructor(ops,weights,config){this.o=ops;this.weights=weights;this.config=config;}
  w(name){const w=this.weights.get(name);if(!w)throw new Error('Missing GPU weight '+name);return w;}
  linear(name,x,options){return this.o.linear(x,this.w(name+'.weight'),this.weights.get(name+'.bias'),options);}
  norm(name,x,channels){const w=this.weights.get(name+'.weight');return this.o.norm(x,channels??w.shape[0],w);}
  normLinear(name,x){return this.o.scope(()=>this.linear(name+'.1',this.norm(name+'.0',x)));}
  transition(name,x){return this.o.scope(()=>{const a=this.norm(name+'.layer_norm_1',x);return this.linear(name+'.linear_3',this.o.mul(this.o.activation(this.linear(name+'.linear_1',a),'silu'),this.linear(name+'.linear_2',a)));});}
  embed(name,f,config){return this.o.scope(()=>{let out;for(const key of Object.keys(config)){const a=this.linear(name+'.embedders.'+key,f[key]);out=out?this.o.add(out,a):a;}return out;});}
  cross(name,q0,kv0,queries,keys,mask){return this.o.scope(()=>{
    const qn=this.norm(name+'.ln_q',q0),kv=this.norm(name+'.ln_kv',kv0);
    const q=this.norm(name+'.q_norm',this.linear(name+'.to_q',qn)),k=this.norm(name+'.k_norm',this.linear(name+'.to_k',kv));
    const v=this.linear(name+'.to_v',kv),gate=this.o.activation(this.linear(name+'.to_g.0',qn),'sigmoid');
    return this.linear(name+'.to_out.0',this.o.attention(q,k,v,{queries,keys,channels:q.length/queries,heads:4,mask,gate}));
  });}
  downcast(name,atoms,tokens,ctx,s){return this.o.scope(()=>{let out=this.o.add(tokens,this.cross(name+'.gca',tokens,atoms,ctx.n,ctx.l,ctx.downMask));if(s)out=this.o.add(out,this.normLinear(name+'.process_s',s));return out;});}
  pairformer(name,s,z,n){return this.o.scope(()=>{
    z=this.o.add(z,this.transition(name+'.z_transition',z));const p=name+'.attention_pair_bias',x=this.norm(p+'.ln_1',s);
    const q=this.linear(p+'.to_q',x),k=this.norm(p+'.to_k.ln',this.linear(p+'.to_k',x)),v=this.norm(p+'.to_v.ln',this.linear(p+'.to_v',x));
    const bias=this.linear(p+'.to_b',this.norm(p+'.ln_0',z)),gate=this.o.activation(this.linear(p+'.to_g.0',x),'sigmoid');
    const out=this.o.attention(q,k,v,{queries:n,keys:n,channels:q.length/n,heads:this.w(p+'.to_b.weight').shape[0],bias,gate,qPreScale:true});
    s=this.o.add(s,this.linear(p+'.to_a',out));s=this.o.add(s,this.transition(name+'.s_transition',s));return [s,z];
  });}
  relative(name,ctx){return this.o.scope(()=>{
    const {n,f,host}=ctx,{r_max:r,s_max:s}=this.config.token_initializer.relative_position_encoding,bins=2*r+3,width=2*bins+2*s+3;
    const cyclic=host.cyclic_asym_ids?.[0]??-1,ring=cyclic<0?0:new Set(Array.from(host.residue_index).filter((_,i)=>host.asym_id[i]===cyclic)).size;
    const data=this.o.kernel([f.asym_id,f.entity_id,f.residue_index,f.token_index,f.sym_id,f.unindexing_pair_mask],n*n*width,`
let pair=idx/${width}u;let i=pair/${n}u;let j=pair%${n}u;let c=idx%${width}u;
let same=a0[i]==a0[j];let entity=a1[i]==a1[j];var offset=i32(a2[i]-a2[j]);
${ring?`if(a0[i]==${cyclic}.0&&a0[j]==${cyclic}.0){if(2*offset>${ring}){offset-=${ring};}else if(2*offset < -${ring}){offset+=${ring};}}`:''}
var residue=select(${2*r+1},clamp(offset+${r},0,${2*r}),same);
var token=select(${2*r+1},clamp(i32(a3[i]-a3[j])+${r},0,${2*r}),same&&a2[i]==a2[j]);
if(a5[pair]!=0.0){residue=${bins-1};token=${bins-1};}
let chain=select(${2*s+1},clamp(i32(a4[i]-a4[j])+${s},0,${2*s}),entity);
out[idx]=select(0.0,1.0,c==u32(residue)||c==${bins}u+u32(token)||(c==${2*bins}u&&entity)||c==${2*bins+1}u+u32(chain));`);
    return this.linear(name+'.linear',data);
  });}
  position(name,pos,mask,sinusoidal=false,addTo){
    const keyA=name+(sinusoidal?'.output_proj':'.process_inverse_dist'),keyB=name+'.process_valid_mask';
    return this.o.fusedPosition(pos,mask,this.w(keyA+'.weight'),this.weights.get(keyA+'.bias'),this.w(keyB+'.weight'),this.weights.get(keyB+'.bias'),{sinusoidal,addTo});
  }
  prepare(host){
    const o=this.o,n=host.residue_index.length,l=host.atom_to_token_map.length,f={};
    for(const [key,data]of Object.entries(host))if(ArrayBuffer.isView(data)||Array.isArray(data))f[key]=o.upload(data);
    const rep=Array.from(host.is_ca).flatMap((v,i)=>v?[i]:[]);if(rep.length!==n)throw new Error('Expected one representative atom per token');
    const repGPU=o.upload(rep),ctx={host,f,n,l,rep:repGPU};
    ctx.downMask=o.kernel([f.atom_to_token_map],n*l,`out[idx]=select(0.0,1.0,u32(a0[idx%${l}u])==idx/${l}u);`);
    const split=this.config.diffusion_module.atom_attention_decoder.upcast.n_split;
    ctx.upMask=o.kernel([f.atom_to_token_map],l*n*split,`out[idx]=select(0.0,1.0,u32(a0[idx/${n*split}u])==(idx%${n*split}u)/${split}u);`);
    ctx.atomForced=this.forced(host,host.atom_to_token_map,this.config.diffusion_module.n_attn_keys,this.config.diffusion_module.n_attn_seq_neighbours);
    ctx.tokenForced=this.forced(host,Int32Array.from({length:n},(_,i)=>i),this.config.diffusion_module.diffusion_transformer.n_keys,this.config.diffusion_module.diffusion_transformer.n_local_tokens);
    return ctx;
  }
  forced(f,tok,keys,near){
    const l=tok.length,n=f.asym_id.length,k=Math.min(keys,l),members=Array.from({length:n},()=>[]);tok.forEach((t,i)=>members[t].push(i));
    const full=Array.from({length:n},()=>new Uint8Array(n)),mask=new Float32Array(l*l);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)full[i][j]=Math.abs(i-j)<=near&&members[i].every(a=>members[j].every(b=>Math.abs(a-b)<=Math.floor(k/2)));
    for(let i=0;i<l;i++)for(let j=0;j<l;j++)mask[i*l+j]=Number(full[tok[i]][tok[j]]&&f.asym_id[tok[i]]===f.asym_id[tok[j]]&&!f.unindexing_pair_mask[tok[i]*n+tok[j]]);
    const result=this.o.upload(mask);
    if(new Set(f.asym_id).size>3){result.multi={};result.multi.chain=this.o.upload(Array.from(tok,t=>f.asym_id[t]));result.multi.local=this.o.upload(Array.from({length:l*l},(_,i)=>Number(Math.abs(tok[Math.floor(i/l)]-tok[i%l])<=near)));result.multi.base=this.o.upload(Array.from({length:l*l},(_,i)=>Number(!f.unindexing_pair_mask[tok[Math.floor(i/l)]*n+tok[i%l]])));}
    return result;
  }
  neighbours(coords,forced,keys){
    const l=coords.length/3,k=Math.min(keys,l);if(forced.multi)return this.multiNeighbours(coords,forced.multi,k);
    if(k===l)return undefined;
    return this.o.kernel([coords,forced],l*l,`let row=idx/${l}u;let col=idx%${l}u;
if(a1[idx]!=0.0){out[idx]=1.0;return;}
var distance=0.0;for(var c=0u;c<3u;c++){let d=a0[row*3u+c]-a0[col*3u+c];distance+=d*d;}
var rank=0u;for(var j=0u;j<${l}u;j++){
 if(a1[row*${l}u+j]!=0.0){rank++;continue;}
 var other=0.0;for(var c=0u;c<3u;c++){let d=a0[row*3u+c]-a0[j*3u+c];other+=d*d;}
 if(other<distance||(other==distance&&j<col)){rank++;}if(rank>=${k}u){break;}}
out[idx]=select(0.0,1.0,rank<${k}u);`,'RFD3 coordinate nearest neighbours');
  }
  multiNeighbours(coords,m,k){return this.o.scope(()=>{
    const o=this.o,l=coords.length/3,inter=Math.max(32,Math.floor(k/4));
    const d=o.kernel([coords],l*l,`let i=idx/${l}u;let j=idx%${l}u;var s=0.0;for(var c=0u;c<3u;c++){let v=a0[i*3u+c]-a0[j*3u+c];s+=v*v;}out[idx]=sqrt(s);`);
    const interScore=o.kernel([d,m.chain,m.base],l*l,`let row=idx/${l}u;let col=idx%${l}u;let v=a0[idx];if(a1[row]==a1[col]||a2[idx]==0.0||v>=12.0){out[idx]=3.4e38;return;}var rank=0u;for(var j=0u;j<${l}u;j++){let other=a0[row*${l}u+j];if(a1[j]==a1[col]&&a2[row*${l}u+j]!=0.0&&(other<v||(other==v&&j<col))){rank++;}}out[idx]=f32(rank)*12.0+v;`);
    const reserved=o.kernel([interScore],l*l,`let row=idx/${l}u;let col=idx%${l}u;let v=a0[idx];if(v==3.4e38){out[idx]=0.0;return;}var rank=0u;for(var j=0u;j<${l}u;j++){let other=a0[row*${l}u+j];if(other<v||(other==v&&j<col)){rank++;}}out[idx]=select(0.0,1.0,rank<${inter}u);`);
    const rowMax=o.kernel([d],l,`var maximum=0.0;for(var j=0u;j<${l}u;j++){maximum=max(maximum,a0[idx*${l}u+j]);}out[idx]=maximum;`);
    const maximum=o.kernel([rowMax],1,`var maximum=0.0;for(var j=0u;j<${l}u;j++){maximum=max(maximum,a0[j]);}out[0]=maximum+1.0;`);
    const intra=o.kernel([d,m.chain,m.base,m.local,reserved,maximum],l*l,`let row=idx/${l}u;let col=idx%${l}u;var v=a0[idx];if(a1[row]!=a1[col]){v+=a5[0];}if(a1[row]==a1[col]&&a3[idx]!=0.0){v=-1.0;}if(a2[idx]==0.0||a4[idx]!=0.0){v=3.4e38;}out[idx]=v;`);
    const ranks=l<=32?o.sortRanks(intra,l):undefined;
    return o.kernel(ranks?[intra,reserved,ranks]:[intra,reserved],l*l,`let row=idx/${l}u;let col=idx%${l}u;let v=a0[idx];var reserved=0u;var rank=0u;for(var j=0u;j<${l}u;j++){reserved+=u32(a1[row*${l}u+j]);let other=a0[row*${l}u+j];if(other<v||(other==v&&j<col)){rank++;}}${ranks?'rank=u32(a2[idx]);':''}out[idx]=a1[idx]+select(0.0,1.0,rank<${k-inter}u||(rank>=${k-inter}u+reserved&&rank<${k}u));`);
  });}
  initialize(ctx){return this.o.scope(()=>{
    const o=this.o,{f,n,l}=ctx,p='token_initializer',cfg=this.config.token_initializer,{c_s:cs,c_z:cz,c_atom:ca,c_atompair:cp}=this.config;
    let s=this.embed(p+'.token_1d_embedder',f,cfg.token_1d_features);s=o.add(s,this.transition(p+'.transition_post_token',s));
    s=this.downcast(p+'.downcast_atom',this.embed(p+'.atom_1d_embedder_1',f,cfg.atom_1d_features),s,ctx);s=o.add(s,this.transition(p+'.transition_post_atom',s));s=this.normLinear(p+'.process_s_init',s);
    const zi=this.linear(p+'.to_z_init_i',s),zj=this.linear(p+'.to_z_init_j',s);
    let z=o.kernel([zi,zj],n*n*cz,`let i=idx/${n*cz}u;let j=(idx/${cz}u)%${n}u;let c=idx%${cz}u;out[idx]=a0[j*${cz}u+c]+a1[i*${cz}u+c];`);
    z=o.add(z,this.relative(p+'.relative_position_encoding',ctx));z=o.add(z,this.linear(p+'.process_token_bonds',f.token_bonds));
    const pos=this.representatives(f.ref_pos,ctx),mask=o.kernel([f.ref_space_uid,ctx.rep],n*n,`out[idx]=select(0.0,1.0,a0[u32(a1[idx/${n}u])]==a0[u32(a1[idx%${n}u])]);`);
    z=o.add(z,this.position(p+'.ref_pos_embedder_tok',pos,mask));
    for(let i=0;i<cfg.n_pairformer_blocks;i++)[s,z]=this.pairformer(p+'.transformer_stack.'+i,s,z,n);
    z=this.normLinear(p+'.process_z_init',o.concat([z,this.relative(p+'.relative_position_encoding2',ctx)],[cz,cz],n*n));
    for(let i=0;i<2;i++)z=o.add(z,this.transition(p+'.transition_1.'+i,z));
    const q=this.embed(p+'.atom_1d_embedder_2',f,cfg.atom_1d_features),st=this.normLinear(p+'.process_s_trunk',s);
    const c=o.kernel([q,st,f.atom_to_token_map],l*ca,`out[idx]=a0[idx]+a1[u32(a2[idx/${ca}u])*${ca}u+idx%${ca}u];`);
    o.free(st);
    let pair=o.scope(()=>{
      const motifMask=o.kernel([f.is_motif_atom_with_fixed_coord],l*l,`out[idx]=a0[idx/${l}u]*a0[idx%${l}u];`);
      return this.position(p+'.motif_pos_embedder',f.motif_pos,motifMask,true);
    });
    {
      const next=o.scope(()=>{
        const refMask=o.kernel([f.ref_space_uid,f.is_motif_atom_with_fixed_seq],l*l,`let i=idx/${l}u;let j=idx%${l}u;out[idx]=select(0.0,a1[i]*a1[j],a0[i]==a0[j]);`);
        return this.position(p+'.ref_pos_embedder',f.ref_pos,refMask,false,pair);
      });
      o.free(pair);pair=next;
    }
    {
      const next=o.scope(()=>{
        const left=this.linear(p+'.process_single_l.1',c,{reluInput:true}),right=this.linear(p+'.process_single_m.1',c,{reluInput:true});
        const zp=this.normLinear(p+'.process_z',z);
        return o.kernel([pair,left,right,zp,f.atom_to_token_map],pair.length,
          `let i=idx/${l*cp}u;let j=(idx/${cp}u)%${l}u;let c=idx%${cp}u;let ti=u32(a4[i]);let tj=u32(a4[j]);out[idx]=(a0[idx]+(a1[i*${cp}u+c]+a2[j*${cp}u+c]))+a3[(ti*${n}u+tj)*${cp}u+c];`);
      });
      o.free(pair);pair=next;
    }
    {
      let mlp=pair;
      for(const index of [1,3,5]){
        const nextMlp=this.linear(p+'.pair_mlp.'+index,mlp,{reluInput:true});
        if(mlp!==pair)o.free(mlp);
        mlp=nextMlp;
      }
      const next=o.add(pair,mlp);
      o.free(pair);o.free(mlp);pair=next;
    }
    {
      const deltaZ=o.scope(()=>{
        const processed=this.linear(p+'.process_pll',pair);
        const temp=o.kernel([processed,f.atom_to_token_map],n*l*cp,`let token=idx/${l*cp}u;let atom=(idx/${cp}u)%${l}u;let c=idx%${cp}u;var sum=0.0;for(var i=0u;i<${l}u;i++){if(u32(a1[i])==token){sum+=a0[(i*${l}u+atom)*${cp}u+c];}}out[idx]=sum;`);
        o.free(processed);
        const pooled=o.kernel([temp,f.atom_to_token_map],n*n*cp,`let i=idx/${n*cp}u;let j=(idx/${cp}u)%${n}u;let c=idx%${cp}u;var sum=0.0;var ci=0.0;var cj=0.0;for(var a=0u;a<${l}u;a++){if(u32(a1[a])==i){ci+=1.0;}if(u32(a1[a])==j){cj+=1.0;sum+=a0[(i*${l}u+a)*${cp}u+c];}}out[idx]=sum/(ci*cj);`);
        o.free(temp);
        return this.linear(p+'.project_pll',pooled);
      });
      const nextZ=o.add(z,deltaZ);
      o.free(z);o.free(deltaZ);z=nextZ;
    }
    return {Q_L_init:q,C_L:c,P_LL:pair,S_I:s,Z_II:z};
  });}
  representatives(coords,ctx){return this.o.kernel([coords,ctx.rep],ctx.n*3,'out[idx]=a0[u32(a1[idx/3u])*3u+idx%3u];');}
  ada(name,a,s,channels){return this.o.scope(()=>{const x=this.o.norm(a,channels),c=this.norm(name+'.ln_s',s);return this.o.add(this.o.mul(this.o.activation(this.linear(name+'.to_gain.0',c),'sigmoid'),x),this.linear(name+'.to_bias',c));});}
  block(name,a,s,z,rows,mask){return this.o.scope(()=>{
    const o=this.o,p=name+'.attention_pair_bias',channels=a.length/rows,x=this.ada(p+'.ada_ln_1',a,s,channels);
    const q=this.norm(p+'.ln_q',this.linear(p+'.to_q',x)),k=this.norm(p+'.ln_k',this.linear(p+'.to_k',x)),v=this.linear(p+'.to_v',x),gate=o.activation(this.linear(p+'.to_g.0',x),'sigmoid'),bias=this.linear(p+'.to_b',z);
    const update=o.attention(q,k,v,{queries:rows,keys:rows,channels,heads:this.w(p+'.to_b.weight').shape[0],mask,gate,bias});
    a=o.add(a,o.mul(this.linear(p+'.to_o',update),o.activation(this.linear(p+'.linear_output_project.0',s),'sigmoid')));
    const tr=name+'.transition_block',norm=this.ada(tr+'.ada_ln',a,s,channels);
    const swiglu=o.mul(o.activation(this.linear(tr+'.linear_1',norm),'silu'),this.linear(tr+'.linear_2',norm));
    return o.add(a,o.mul(o.activation(this.linear(tr+'.linear_output_project.0',s),'sigmoid'),this.linear(tr+'.linear_3',swiglu)));
  });}
  time(index,mask,t){return this.o.scope(()=>{
    const p='diffusion_module',w=this.w(p+'.fourier_embedding.'+index+'.w'),b=this.w(p+'.fourier_embedding.'+index+'.b'),channels=w.shape[0],time=Math.fround(Math.fround(Math.log(Math.fround(Math.max(t,1e-20)/16)))/4);
    const fourier=this.o.kernel([w,b],mask.length*channels,`out[idx]=cos(6.2831854820251465*(${time.toPrecision(12)}*a0[idx%${channels}u]+a1[idx%${channels}u]));`);
    const projected=this.normLinear(p+'.process_n.'+index,fourier),c=projected.length/mask.length;
    return this.o.kernel([projected,mask],projected.length,`out[idx]=select(a0[idx],0.0,a1[idx/${c}u]!=0.0||${t===0?'true':'false'});`);
  });}
  distogram(coords,ctx,sigma){return this.o.scope(()=>{
    const pos=this.representatives(coords,ctx),n=ctx.n;
    return this.o.kernel([pos],n*n*65,`let pair=idx/65u;let i=pair/${n}u;let j=pair%${n}u;var square=0.0;for(var c=0u;c<3u;c++){let d=a0[i*3u+c]-a0[j*3u+c];square+=d*d;}
let distance=sqrt(square);var bin=0u;for(var b=0u;b<64u;b++){if(distance>${1/sigma}+f32(b)*${Math.fround((30/sigma-1/sigma)/63)}){bin++;}}out[idx]=select(0.0,1.0,idx%65u==bin);`);
  });}
  denoise(ctx,input,t,recycles,init){return this.o.scope(()=>{
    const o=this.o,{f,n,l}=ctx,cfg=this.config.diffusion_module,p='diffusion_module',ct=cfg.c_token,cz=this.config.c_z,sigma=cfg.sigma_data;
    const uniform=o.kernel([input],input.length,`out[idx]=a0[idx]/sqrt(${t*t+sigma*sigma});`);
    const scaled=o.kernel([input,f.is_motif_atom_with_fixed_coord],input.length,`let time=select(${t},0.0,a1[idx/3u]!=0.0);out[idx]=a0[idx]/sqrt(time*time+${sigma*sigma}.0);`);
    const atomMask=this.neighbours(input,ctx.atomForced,cfg.n_attn_keys);
    const projected=this.linear(p+'.process_a.linear',scaled);
    const a0=o.kernel([projected,f.atom_to_token_map],n*ct,`let token=idx/${ct}u;let c=idx%${ct}u;var sum=0.0;var count=0.0;for(var a=0u;a<${l}u;a++){if(u32(a1[a])==token){sum+=a0[a*${ct}u+c];count+=1.0;}}out[idx]=sum/count;`);
    let c=o.add(init.C_L,this.time(0,f.is_motif_atom_with_fixed_coord,t));c=o.add(c,this.normLinear(p+'.process_c',c));
    const baseS=o.add(this.downcast(p+'.downcast_c',init.C_L,init.S_I,ctx),this.time(1,f.is_motif_token_with_fully_fixed_coord,t));
    let q=o.add(init.Q_L_init,this.linear(p+'.process_r',scaled));
    for(let b=0;b<cfg.atom_attention_encoder.n_blocks;b++)q=this.block(p+'.encoder.blocks.'+b,q,c,init.P_LL,l,atomMask);
    const baseQ=q,baseA=this.downcast(p+'.downcast_q',q,a0,ctx,baseS);
    let last,lastDistance,logits;
    for(let cycle=0;cycle<recycles;cycle++){
      const result=o.scope(()=>{
        let s=baseS,z=init.Z_II,a=baseA,q=baseQ;const enc=p+'.diffusion_token_encoder',ecfg=cfg.diffusion_token_encoder;
        for(let b=0;b<2;b++)s=o.add(s,this.transition(enc+'.transition_1.'+b,s));
        const parts=[z],widths=[cz];if(ecfg.use_distogram){parts.push(this.distogram(uniform,ctx,sigma));widths.push(65);}if(ecfg.use_self){parts.push(lastDistance??o.fill(n*n*65));widths.push(65);}
        z=this.normLinear(enc+'.process_z',o.concat(parts,widths,n*n));for(let b=0;b<2;b++)z=o.add(z,this.transition(enc+'.transition_2.'+b,z));
        for(let b=0;b<ecfg.n_pairformer_blocks;b++)[s,z]=this.pairformer(enc+'.pairformer_stack.'+b,s,z,n);
        const rawTokenMask=this.neighbours(this.representatives(last??input,ctx),ctx.tokenForced,cfg.diffusion_transformer.n_keys);
        const tokenMask=rawTokenMask?o.kernel([rawTokenMask],rawTokenMask.length,'out[idx]=min(a0[idx],1.0);'):undefined;
        for(let b=0;b<cfg.diffusion_transformer.n_block;b++)a=this.block(p+'.diffusion_transformer.blocks.'+b,a,s,z,n,tokenMask);
        const split=cfg.atom_attention_decoder.upcast.n_split;
        for(let b=0;b<cfg.atom_attention_decoder.n_blocks;b++){
          q=o.add(q,this.cross(p+'.decoder.upcast.'+b+'.gca',q,a,l,n*split,ctx.upMask));q=this.block(p+'.decoder.atom_transformer.'+b,q,c,init.P_LL,l,atomMask);
        }
        a=this.downcast(p+'.decoder.downcast',q,a,ctx,s);
        const update=this.normLinear(p+'.to_r_update',q);
        const coords=o.kernel([update,input,f.is_motif_atom_with_fixed_coord],input.length,`let time=select(${t},0.0,a2[idx/3u]!=0.0);let denom=${sigma*sigma}.0+time*time;out[idx]=(${sigma*sigma}.0/denom)*a1[idx]+(${sigma}.0*time/sqrt(denom))*a0[idx];`);
        return {coords,distance:this.distogram(coords,ctx,1),logits:this.linear(p+'.sequence_head.linear',a)};
      });
      if(last)o.free(last);if(lastDistance)o.free(lastDistance);if(logits)o.free(logits);
      last=result.coords;lastDistance=result.distance;logits=result.logits;
    }
    const valid=this.config.sequence_valid_out_mask;
    // WebGPU permits flushing subnormals. Decode the FP32 softmax's positive
    // bit ordering explicitly, preserving PyTorch's all-zero/underflow ties.
    const indices=o.kernel([logits],n,`var maximum=-3.4e38;for(var c=0u;c<32u;c++){maximum=max(maximum,a0[idx*32u+c]);}var sum=0.0;for(var c=0u;c<32u;c++){sum+=exp(a0[idx*32u+c]-maximum);}var best=0u;var selected=0u;
for(var c=0u;c<32u;c++){if(${valid.map((v,i)=>v?`c==${i}u`:null).filter(Boolean).join('||')}){let delta=a0[idx*32u+c]-maximum;var bits=0u;if(delta-log(sum)< -87.33654475){bits=u32(round(exp(delta+103.2789299)/sum));}else{bits=bitcast<u32>(exp(delta)/sum);}if(bits>best){best=bits;selected=c;}}}out[idx]=f32(selected);`);
    return {X_L:last,sequence_logits_I:logits,sequence_indices_I:indices};
  });}
}
