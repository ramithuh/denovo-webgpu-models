import {geometry} from './geometry.js';
export class CPUOps {
 constructor(weights){this.weights=weights;}
 tensor(data,shape){return {data,shape};}
 async read(x){return x.data;}
 keep(){}
 async checkpoint(){}
 linear(x,p){
  const w=this.weights[p+'.weight'],b=this.weights[p+'.bias'],[O,I]=w.shape,M=x.data.length/I,y=new Float32Array(M*O),a=x.data,v=w.data;
  for(let r=0;r<M;r++)for(let o=0;o<O;o++){
    let s0=0,s1=0,s2=0,s3=0,k=0,ai=r*I,wi=o*I;
    for(;k+3<I;k+=4){s0+=a[ai+k]*v[wi+k];s1+=a[ai+k+1]*v[wi+k+1];s2+=a[ai+k+2]*v[wi+k+2];s3+=a[ai+k+3]*v[wi+k+3];}
    let s=s0+s1+s2+s3;for(;k<I;k++)s+=a[ai+k]*v[wi+k];y[r*O+o]=s+(b?.data[o]??0);
  }
  return this.tensor(y,[M,O]);
 }
 add(a,b){return this.tensor(Float32Array.from(a.data,(v,i)=>v+b.data[i]),a.shape);}
 relu(a){return this.tensor(Float32Array.from(a.data,v=>Math.max(0,v)),a.shape);}
 concat(xs){const M=xs[0].shape[0],widths=xs.map(x=>x.data.length/M),D=widths.reduce((a,b)=>a+b,0),y=new Float32Array(M*D);for(let r=0;r<M;r++){let off=r*D;for(let j=0;j<xs.length;j++){y.set(xs[j].data.subarray(r*widths[j],(r+1)*widths[j]),off);off+=widths[j];}}return this.tensor(y,[M,D]);}
 slice(a,start,width){const [M,D]=a.shape,y=new Float32Array(M*width);for(let r=0;r<M;r++)y.set(a.data.subarray(r*D+start,r*D+start+width),r*width);return this.tensor(y,[M,width]);}
 norm(x,p,dihedral=false){const [M,D]=x.shape,y=new Float32Array(x.data.length),w=this.weights[p+(dihedral?'.gain':'.weight')].data,b=this.weights[p+'.bias'].data;
  for(let r=0;r<M;r++){let mean=0;for(let j=0;j<D;j++)mean+=x.data[r*D+j];mean/=D;let v=0;for(let j=0;j<D;j++)v+=(x.data[r*D+j]-mean)**2;const den=dihedral?Math.sqrt(v/(D-1)+1e-6)+1e-6:Math.sqrt(v/D+1e-5);for(let j=0;j<D;j++)y[r*D+j]=(x.data[r*D+j]-mean)/den*w[j]+b[j];}return this.tensor(y,x.shape);
 }
 transposeVectors(x,reverse=false){const M=reverse?x.shape[0]/3:x.shape[0],V=reverse?x.shape[1]:x.shape[1]/3,y=new Float32Array(x.data.length);for(let r=0;r<M;r++)for(let v=0;v<V;v++)for(let k=0;k<3;k++){const a=(r*V+v)*3+k,b=(r*3+k)*V+v;if(reverse)y[a]=x.data[b];else y[b]=x.data[a];}return this.tensor(y,reverse?[M,V*3]:[M*3,V]);}
 vectorLength(x){const [M,D]=x.shape,V=D/3,y=new Float32Array(M*V);for(let i=0;i<y.length;i++)y[i]=Math.sqrt(x.data[i*3]**2+x.data[i*3+1]**2+x.data[i*3+2]**2+1e-8);return this.tensor(y,[M,V]);}
 vectorGate(v,g){return this.tensor(Float32Array.from(v.data,(x,i)=>x/(1+Math.exp(-g.data[Math.floor(i/3)]))),v.shape);}
 vectorNorm(x){const [M,D]=x.shape,V=D/3,y=new Float32Array(x.data.length);for(let r=0;r<M;r++){let sum=0,count=0;const norms=[];for(let j=0;j<V;j++){const i=r*D+j*3,n=x.data[i]**2+x.data[i+1]**2+x.data[i+2]**2+1e-4;norms.push(n);if(n>2e-4){sum+=n;count++;}}const den=Math.sqrt(sum/(1e-4+count)+1e-4);for(let j=0;j<V;j++)for(let k=0;k<3;k++)y[r*D+j*3+k]=norms[j]>2e-4?x.data[r*D+j*3+k]/den:0;}return this.tensor(y,x.shape);}
 rotate(x,frames){const [M,D]=x.shape,y=new Float32Array(x.data.length);for(let r=0;r<M;r++)for(let j=0;j<D/3;j++)for(let k=0;k<3;k++){let s=0;for(let z=0;z<3;z++)s+=x.data[r*D+j*3+z]*frames.data[r*9+k*3+z];y[r*D+j*3+k]=s;}return this.tensor(y,x.shape);}
 gather(x,index,which){const E=index.shape[0],D=x.shape[1],y=new Float32Array(E*D);for(let e=0;e<E;e++){const r=index.data[e*2+which];if(r>=0)y.set(x.data.subarray(r*D,(r+1)*D),e*D);}return this.tensor(y,[E,D]);}
 aggregate(x,index,N){const D=x.shape[1],y=new Float32Array(N*D),counts=new Int32Array(N);for(let e=0;e<index.shape[0];e++){const dst=index.data[e*2+1];if(dst<0)continue;counts[dst]++;for(let j=0;j<D;j++)y[dst*D+j]=Math.fround(y[dst*D+j]+x.data[e*D+j]);}for(let r=0;r<N;r++)for(let j=0;j<D;j++)y[r*D+j]/=Math.max(counts[r],1);return this.tensor(y,[N,D]);}
 embed(tokens,p){const w=this.weights[p+'.weight'],D=w.shape[1],M=tokens.shape[0],y=new Float32Array(M*D);for(let r=0;r<M;r++)for(let j=0;j<D;j++)y[r*D+j]=Math.fround(w.data[tokens.data[r]*D+j]*Math.sqrt(D));return this.tensor(y,[M,D]);}
 position(x,padding,zeroPad=false){const [M,D]=x.shape,y=new Float32Array(x.data.length);for(let r=0;r<M;r++)for(let j=0;j<D;j++){const angle=Math.fround((r+2)*Math.fround(Math.exp(Math.fround((j%(D/2))*-Math.log(10000)/(D/2-1))))),pos=padding.data[r]?0:Math.fround(j<D/2?Math.sin(angle):Math.cos(angle));y[r*D+j]=zeroPad&&padding.data[r]?0:Math.fround(x.data[r*D+j]+pos);}return this.tensor(y,x.shape);}
 attention(q,k,v,padding,causal=false){const M=q.shape[0],S=k.shape[0],D=q.shape[1],H=8,C=D/H,y=new Float32Array(M*D),scores=new Float32Array(S);for(let r=0;r<M;r++)for(let h=0;h<H;h++){let max=-Infinity;for(let s=0;s<S;s++){let dot=0;for(let j=0;j<C;j++)dot+=q.data[r*D+h*C+j]*k.data[s*D+h*C+j];scores[s]=padding.data[s]||(causal&&s>r)?-Infinity:dot/Math.sqrt(C);max=Math.max(max,scores[s]);}let sum=0;for(let s=0;s<S;s++){scores[s]=Math.exp(scores[s]-max);sum+=scores[s];}for(let j=0;j<C;j++){let out=0;for(let s=0;s<S;s++)out+=scores[s]/sum*v.data[s*D+h*C+j];y[r*D+h*C+j]=out;}}return this.tensor(y,[M,D]);}
 features(input){const g=geometry(input),N=input.N,E=g.E;return Object.fromEntries(Object.entries(g).filter(([k])=>k!=='E').map(([k,v])=>[k,this.tensor(v,[k.startsWith('edge')?E:N,k==='edgeIndex'?2:v.length/(k.startsWith('edge')?E:N)])]));}
}
