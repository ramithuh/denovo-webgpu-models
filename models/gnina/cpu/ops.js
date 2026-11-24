import {tensorToFloat} from "../shared/loader.js";
function tensor(data,c,d){return {data,c,d};}
export class CPUOps {
  constructor(tensors) {
    this.weights=Object.fromEntries(Object.entries(tensors).map(([key,t])=>[key,{data:tensorToFloat(t),shape:t.shape}]));
  }
  read(x){return x.data;}
  readMany(xs){return xs.map(x=>x.data);}
  release(){}
  pool({data,c,d},kind,k) {
    const n=d/k, out=new Float32Array(c*n**3);
    let i=0;
    for(let ch=0;ch<c;ch++) for(let x=0;x<d;x+=k) for(let y=0;y<d;y+=k) for(let z=0;z<d;z+=k) {
      let v=kind==="max"?-Infinity:0;
      for(let a=0;a<k;a++) for(let b=0;b<k;b++) for(let e=0;e<k;e++) {
        const f=data[((ch*d+x+a)*d+y+b)*d+z+e];
        v=kind==="max"?Math.max(v,f):v+f;
      }
      out[i++]=kind==="max"?v:v/k**3;
    }
    return tensor(out,c,n);
  }
  norm({data,c,d},name) {
    const w=s=>this.weights[name+"."+s].data, out=new Float32Array(data.length);
    const mean=w("running_mean"),variance=w("running_var"),scale=w("weight"),bias=w("bias"),v=d**3;
    for(let ch=0;ch<c;ch++) {
      const m=mean[ch], a=scale[ch]/Math.sqrt(variance[ch]+1e-5), b=bias[ch];
      for(let i=ch*v;i<(ch+1)*v;i++) out[i]=(data[i]-m)*a+b;
    }
    return tensor(out,c,d);
  }
  concat(x,y) {
    const out=new Float32Array(x.data.length+y.data.length);
    out.set(x.data);out.set(y.data,x.data.length);
    return tensor(out,x.c+y.c,x.d);
  }
  conv({data,c,d},name) {
    const {data:w,shape:[oc,,k]}=this.weights[name+".weight"],bias=this.weights[name+".bias"].data;
    const v=d**3, inner=c*k**3, pad=k>>1, out=new Float32Array(oc*v);
    // Lower eight adjacent output voxels at a time so each weight feeds eight accumulators.
    const tile=new Float32Array(inner*8);
    for(let pos=0;pos<v;pos+=8) {
      for(let lane=0;lane<8;lane++) {
        const p=pos+lane, x=Math.floor(p/(d*d)),y=Math.floor(p/d)%d,z=p%d;
        let t=lane;
        for(let ch=0;ch<c;ch++) for(let a=x-pad;a<=x+pad;a++) for(let b=y-pad;b<=y+pad;b++) for(let e=z-pad;e<=z+pad;e++,t+=8)
          tile[t]=a>=0&&a<d&&b>=0&&b<d&&e>=0&&e<d?data[((ch*d+a)*d+b)*d+e]:0;
      }
      for(let ch=0;ch<oc;ch++) {
        let a=0,b=0,e=0,f=0,g=0,h=0,j=0,l=0;
        for(let u=0,t=0,wi=ch*inner;u<inner;u++,t+=8) {
          const m=w[wi+u];
          a+=m*tile[t];b+=m*tile[t+1];e+=m*tile[t+2];f+=m*tile[t+3];
          g+=m*tile[t+4];h+=m*tile[t+5];j+=m*tile[t+6];l+=m*tile[t+7];
        }
        const offset=ch*v+pos,bs=bias[ch];
        out[offset]=Math.max(0,a+bs);out[offset+1]=Math.max(0,b+bs);out[offset+2]=Math.max(0,e+bs);out[offset+3]=Math.max(0,f+bs);
        out[offset+4]=Math.max(0,g+bs);out[offset+5]=Math.max(0,h+bs);out[offset+6]=Math.max(0,j+bs);out[offset+7]=Math.max(0,l+bs);
      }
    }
    return tensor(out,oc,d);
  }
  heads(x,pose,affinity) {
    const linear=(name,row)=>{
      const w=this.weights[name+".weight"].data;
      let sum=this.weights[name+".bias"].data[row];
      for(let i=0;i<x.data.length;i++) sum+=x.data[i]*w[row*x.data.length+i];
      return sum;
    };
    return tensor(Float32Array.of(1/(1+Math.exp(linear(pose,0)-linear(pose,1))),linear(affinity,0)),2,1);
  }
}
