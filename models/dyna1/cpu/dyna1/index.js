import { tokenize, decodeTensor, loadWeights } from '../../shared/dyna1/index.js';

export function linear(x,w,b,rows,dim,outDim) {
  const y = new Float32Array(rows*outDim);
  for(let r=0;r<rows;r++) for(let o=0;o<outDim;o++) {
    let s0=0,s1=0,s2=0,s3=0,k=0;const xi=r*dim,wi=o*dim;
    for(;k+3<dim;k+=4) {s0+=x[xi+k]*w[wi+k];s1+=x[xi+k+1]*w[wi+k+1];s2+=x[xi+k+2]*w[wi+k+2];s3+=x[xi+k+3]*w[wi+k+3];}
    let s=s0+s1+s2+s3;for(;k<dim;k++) s+=x[xi+k]*w[wi+k];y[r*outDim+o]=s+(b?.[o]??0);
  }
  return y;
}
export function layerNorm(x,w,b,rows,dim,eps=1e-5) {
  const y=new Float32Array(x.length);
  for(let r=0;r<rows;r++) {const off=r*dim;let mean=0,v=0;for(let j=0;j<dim;j++) mean+=x[off+j];mean/=dim;for(let j=0;j<dim;j++) v+=(x[off+j]-mean)**2;const inv=1/Math.sqrt(v/dim+eps);for(let j=0;j<dim;j++) y[off+j]=(x[off+j]-mean)*inv*w[j]+(b?.[j]??0);}
  return y;
}
function rope(x,L,D,H) {
  const dh=D/H,half=dh/2,out=new Float32Array(x.length);
  for(let p=0;p<L;p++) for(let h=0;h<H;h++) for(let j=0;j<half;j++) {
    const a=Math.fround(p*Math.fround(1/10000**(2*j/dh))),c=Math.fround(Math.cos(a)),s=Math.fround(Math.sin(a)),i=p*D+h*dh+j;
    out[i]=Math.fround(x[i]*c)-Math.fround(x[i+half]*s);out[i+half]=Math.fround(x[i+half]*c)+Math.fround(x[i]*s);
  }
  return out;
}
function attention(q,k,v,L,D,H) {
  const dh=D/H,y=new Float32Array(L*D),p=new Float32Array(L);
  for(let h=0;h<H;h++) for(let i=0;i<L;i++) {
    let max=-Infinity;
    for(let j=0;j<L;j++) {let s=0;for(let d=0;d<dh;d++) s+=q[i*D+h*dh+d]*k[j*D+h*dh+d];p[j]=s/Math.sqrt(dh);max=Math.max(max,p[j]);}
    let sum=0;for(let j=0;j<L;j++) {p[j]=Math.exp(p[j]-max);sum+=p[j];}for(let j=0;j<L;j++) p[j]/=sum;
    for(let d=0;d<dh;d++) {let s=0;for(let j=0;j<L;j++) s+=p[j]*v[j*D+h*dh+d];y[i*D+h*dh+d]=s;}
  }
  return y;
}
function residual(x,y,scale=1) {const out=new Float32Array(x.length);for(let i=0;i<x.length;i++) out[i]=x[i]+Math.fround(y[i]/scale);return out;}
export class Dyna1CPU {
  constructor(bundle) {this.config=bundle.config;this.w={};for(const [k,v] of Object.entries(bundle.weights)) this.w[k]=v instanceof Float32Array?v:decodeTensor(v);}
  predict(sequence,{returnIntermediates=false,signal}={}) {
    const start=performance.now(),{tokens,residues}=tokenize(sequence,this.config.max_sequence_length),L=tokens.length,D=this.config.d_model,w=this.w,traces={tokens};
    const save=(n,v)=>{if(returnIntermediates) traces[n]=v;};
    const ln=(x,p,eps=1e-5)=>layerNorm(x,w[p+'.weight'],w[p+'.bias'],L,D,eps);
    const lin=(x,p,di=D,do_=D)=>linear(x,w[p+'.weight'],w[p+'.bias'],L,di,do_);
    const split=(qkv)=>{const q=new Float32Array(L*D),k=new Float32Array(L*D),v=new Float32Array(L*D);for(let i=0;i<L;i++){q.set(qkv.subarray(i*3*D,i*3*D+D),i*D);k.set(qkv.subarray(i*3*D+D,i*3*D+2*D),i*D);v.set(qkv.subarray(i*3*D+2*D,(i+1)*3*D),i*D);}return [q,k,v];};
    let x=new Float32Array(L*D);for(let i=0;i<L;i++) x.set(w['embedding.weight'].subarray(tokens[i]*D,(tokens[i]+1)*D),i*D);save('embedding',x);
    for(let i=0;i<this.config.backbone_layers;i++) {
      signal?.throwIfAborted();const p=`backbone.${i}`,F=this.config.ffn_size,H=this.config.backbone_heads;
      let [q,k,v]=split(lin(ln(x,p+'.attn.layernorm_qkv.0'),p+'.attn.layernorm_qkv.1',D,3*D));
      q=rope(ln(q,p+'.attn.q_ln'),L,D,H);k=rope(ln(k,p+'.attn.k_ln'),L,D,H);
      x=residual(x,lin(attention(q,k,v,L,D,H),p+'.attn.out_proj'),this.config.residual_scale);
      const a=lin(ln(x,p+'.ffn.0'),p+'.ffn.1',D,2*F),act=new Float32Array(L*F);
      for(let r=0;r<L;r++) for(let j=0;j<F;j++) {const z=a[r*2*F+j];act[r*F+j]=Math.fround(z/(1+Math.exp(-z)))*a[r*2*F+F+j];}
      x=residual(x,lin(act,p+'.ffn.3',F,D),this.config.residual_scale);save(`backbone.${i}`,x);
    }
    save('final_embeddings',x);x=ln(x,'head.norm1',1e-12);
    for(let i=0;i<this.config.head_layers;i++) {
      signal?.throwIfAborted();const p=`head.encoder.layers.${i}`,F=this.config.head_ffn_size;
      const [q,k,v]=split(linear(ln(x,p+'.norm1'),w[p+'.self_attn.in_proj_weight'],w[p+'.self_attn.in_proj_bias'],L,D,3*D));
      x=residual(x,lin(attention(q,k,v,L,D,this.config.head_heads),p+'.self_attn.out_proj'));
      const a=lin(ln(x,p+'.norm2'),p+'.linear1',D,F);for(let j=0;j<a.length;j++) a[j]=Math.max(0,a[j]);
      x=residual(x,lin(a,p+'.linear2',F,D));save(`head.${i}`,x);
    }
    x=ln(x,'head.norm2',1e-12);
    const logits=lin(x,'head.output_layer',D,1),probs=Float32Array.from(logits,z=>1/(1+Math.exp(-(Math.fround(z-Math.fround(Math.log(.05/.95)))))));
    save('logits',logits);save('probs',probs);
    return {logits,probs,residues,timingMs:performance.now()-start,...(returnIntermediates?{intermediates:traces}:{})};
  }
  destroy() {this.w={};}
}
export async function createDyna1CPU(options) {return new Dyna1CPU(await loadWeights(options.baseUrl,options));}
