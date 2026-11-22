"""Render measured tables from successful verification artifacts; no invented metrics."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def sci(x):return f'{x:.3e}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 export=read('fixtures/export_metrics.json');bundles={p:read(f'weights/esmc-300m/{p}/manifest.json') for p in ['fp32','fp16','int8']}
 cpu=read('fixtures/cpu_metrics.json');gpu={p:read(f'fixtures/gpu_{p}_metrics.json') for p in ['fp32','fp16','int8','fp16_native']}
 rows=[]
 for p,m in bundles.items():rows.append([p.upper(),len(m['shards']),f"{m['total_bytes']:,}",sci(m['weight_rel_rms'])])
 exports=table(['Bundle','Shards','Bytes including headers','Weight relRMS'],rows)+'\n\n'
 rows=[]
 for p in ['fp16','int8']:
  for name in ['trp_cage','ubiquitin']:
   rows.append([p.upper(),name]+[sci(export[p][name+'/'+key]['relRms']) for key in ['embeddings','wt_logits','wt_llr','masked_llr','fill_logits','naturalness','pseudo_perplexity']])
 exports+='Numerical quantization error against original FP32 (FP32 round trip is exactly zero in every mode):\n\n'+table(['Precision','Protein','Embeddings','MLM logits','WT LLR','Masked LLR','Fill logits','Naturalness','Pseudo-perplexity'],rows)
 sources=[cpu]+[gpu[p]['metrics'] for p in gpu]
 groups=[('Embeddings, all 30 blocks + final LN',lambda k:k.startswith(('wt/block.','wt/embedding','wt/final_ln','wt/residue_embeddings','wt/pooled'))),('MLM logits and head stages',lambda k:k.startswith(('wt/logits','wt/head.'))),('Wild-type L×20 LLR',lambda k:k=='wt/llr'),('Masked L×20 LLR + every sweep block',lambda k:k.startswith('masked/') and not any(x in k for x in ['naturalness','pseudo_perplexity','wt_logp'])),('Naturalness and pseudo-perplexity',lambda k:k in ['masked/wt_logp','masked/naturalness','masked/pseudo_perplexity']),('Masked fill, all blocks + probabilities',lambda k:k.startswith('fill/')),('Attention maps + padding/packed inputs',lambda k:k.startswith('wt/attention.'))]
 rows=[]
 for label,predicate in groups:
  values=[]
  for source in sources:
   stats=[s['relRms'] for n in ['trp_cage','ubiquitin'] for k,s in source[n]['tensors'].items() if predicate(k)]
   if label.startswith('Attention'):stats += [s['relRms'] for s in source['tokens'].values()]
   v=max(stats);assert v<1e-4,(label,v);values.append(sci(v))
  rows.append([label]+values)
 modes='All entries passed on both proteins. Entries are the **maximum measured relRMS** across the corresponding traces/outputs.\n\n'+table(['Mode / coverage','CPU FP32','GPU FP32','GPU FP16 core','GPU INT8','GPU FP16 native'],rows)
 timing=table(['Storage / loader','Cold load (ms)','Trp-cage warm (ms)','Ubiquitin warm (ms)'],[[p,f"{g['loadMs']:.1f}",f"{g['metrics']['trp_cage']['medianMs']:.1f}",f"{g['metrics']['ubiquitin']['medianMs']:.1f}"] for p,g in gpu.items()])
 p=ROOT/'README_WEBGPU.md';text=p.read_text()
 for key,value in [('EXPORT',exports),('MODE',modes),('TIMING',timing)]:
  marker=f'<!-- GENERATED_{key}_TABLE -->';end=f'<!-- END_{key}_TABLE -->'
  text=re.sub(re.escape(marker)+r'(?:.*?'+re.escape(end)+r')?',lambda _:marker+'\n\n'+value+'\n\n'+end,text,flags=re.S)
 p.write_text(text)
if __name__=='__main__':main()
