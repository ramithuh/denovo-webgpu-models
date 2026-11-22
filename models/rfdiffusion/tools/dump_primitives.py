from reference import *
from rfdiffusion.util import rigid_from_3_points
from e3nn.o3 import spherical_harmonics
import json
out={}
torch.manual_seed(1234)
for n in [1,2,3,8,24,32,64,128,256]:
 x=torch.randn(128,n)
 out['norm'+str(n)]={'x':x.tolist(),'y':torch.linalg.vector_norm(x,dim=-1).tolist()}
 if n>2:out['softmax'+str(n)]={'x':x.tolist(),'y':x.softmax(-1).tolist()}
x=torch.randn(128,3);out['sh']={'x':x.tolist(),'y':spherical_harmonics([0,1,2],x,normalize=True).tolist()}
x=torch.randn(128,3);b=torch.randn(128,3);out['cross']={'x':x.tolist(),'b':b.tolist(),'y':torch.cross(x,b,dim=-1).tolist()}
x=torch.randn(128,3,3);b=torch.randn_like(x);out['matmul3']={'x':x.tolist(),'b':b.tolist(),'y':torch.einsum('nij,njk->nik',x,b).tolist()}
x=torch.randn(128,64);w=torch.randn(64);b=torch.randn(64);out['group']={'x':x.tolist(),'w':w.tolist(),'b':b.tolist(),'y':torch.nn.functional.group_norm(x,2,w,b).tolist()}
x=torch.randn(24,24,8);out['softmaxAxis']={'x':x.tolist(),'y':x.softmax(1).tolist()}
# DGL incoming edge order is source-major, then destination, then head.
import dgl
from dgl.nn.functional import edge_softmax
n=24
src=torch.arange(n).repeat_interleave(n);dst=torch.arange(n).repeat(n)
g=dgl.graph((src,dst),num_nodes=n)
x=torch.randn(n,n,4)*12
out['dglSoftmax']={'x':x.tolist(),'y':edge_softmax(g,x.reshape(n*n,4)).reshape(n,n,4).tolist()}
x=torch.randn(512,3)*.03
norm=torch.sqrt(1+torch.sum(x*x,dim=-1,keepdim=True));q=torch.cat((torch.ones(512,1),x),-1)/norm
q=q[:,:,None]*q[:,None,:]
r=torch.stack((q[:,0,0]+q[:,1,1]-q[:,2,2]-q[:,3,3],2*q[:,1,2]-2*q[:,0,3],2*q[:,1,3]+2*q[:,0,2],2*q[:,1,2]+2*q[:,0,3],q[:,0,0]-q[:,1,1]+q[:,2,2]-q[:,3,3],2*q[:,2,3]-2*q[:,0,1],2*q[:,1,3]-2*q[:,0,2],2*q[:,2,3]+2*q[:,0,1],q[:,0,0]-q[:,1,1]-q[:,2,2]+q[:,3,3]),-1).reshape(512,3,3)
out['quaternion']={'x':x.tolist(),'y':r.tolist()}
# Exact FMA midpoint cases: evaluating a*b+c in F64 then F32 double-rounds.
a=torch.tensor([1+2**-12,-1-2**-12,1+2**-12,-1-2**-12]);b=torch.full_like(a,1+2**-12);c=torch.tensor([2**-60,-2**-60,-2**-60,2**-60])
y=torch.stack([torch.nn.functional.linear(a[i:i+1],b[i:i+1,None],c[i:i+1])[0] for i in range(4)])
out['fma']={'x':a.tolist(),'b':b.tolist(),'c':c.tolist(),'y':y.tolist()}
x=torch.tensor([1.25,-1.25,1.0833343267440796,1.4166686534881592]);out['scalarScale']={'x':x.tolist(),'scale':0.01,'y':(x*0.01).tolist()}
(ROOT/'.cache/fixtures/primitives.json').write_text(json.dumps(out))
(ROOT/'.cache/primitives.json').write_text(json.dumps(out))
