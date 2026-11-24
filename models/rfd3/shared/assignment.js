// Rectangular minimum-cost assignment by shortest augmenting paths (rows <= columns).
export function linearAssignment(cost){
 const n=cost.length,m=cost[0]?.length;if(!n||!m||n>m||cost.some(r=>r.length!==m||r.some(x=>Number.isNaN(x))))throw new Error('Invalid assignment dimensions');
 const u=new Float64Array(n+1),v=new Float64Array(m+1),p=new Int32Array(m+1),way=new Int32Array(m+1);
 for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=new Float64Array(m+1).fill(Infinity),used=new Uint8Array(m+1);
  do{used[j0]=1;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=m;j++)if(!used[j]){const cur=cost[i0-1][j-1]-u[i0]-v[j];if(cur<min[j]){min[j]=cur;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}
   if(!Number.isFinite(delta))throw new Error('No feasible atom-name assignment for unindexed motif');
   for(let j=0;j<=m;j++)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else min[j]-=delta;j0=j1;
  }while(p[j0]!==0);
  do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);
 }
 const result=new Int32Array(n);for(let j=1;j<=m;j++)if(p[j])result[p[j]-1]=j-1;return result;
}
