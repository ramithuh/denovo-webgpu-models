// De novo canonical protein Atom14 features, following pinned RFD3's
// PadTokensWithVirtualAtoms and CreateDesignReferenceFeatures. No fixture reads.
export function createMonomerFeatures({ length, cyclic = false } = {}) {
  if (!Number.isSafeInteger(length) || length < 1) throw new RangeError('length must be a positive integer');
  if (typeof cyclic !== 'boolean') throw new TypeError('cyclic must be boolean');
  if (cyclic && length < 3) throw new RangeError('A cyclic peptide needs at least three residues');
  const n=length,l=n*14,f={};
  if (!Number.isSafeInteger(n*n*128) || n*n*128 > 2**30) throw new RangeError('Requested feature arrays are too large');
  const zeros=(name,count)=>f[name]=new Float32Array(count);
  for(const name of ['active_acceptor','active_donor','has_zero_occupancy','is_atom_level_hotspot','is_motif_atom_unindexed',
    'is_motif_atom_with_fixed_coord','is_motif_atom_with_fixed_seq','ref_charge','ref_is_motif_atom_unindexed',
    'ref_is_motif_atom_with_fixed_coord','ref_mask','ref_pos_is_ground_truth'])zeros(name,l);
  for(const name of ['asym_id','entity_id','sym_id','is_dna','is_ligand','is_motif_token_unindexed',
    'is_motif_token_with_fully_fixed_coord','is_non_loopy','is_polar','is_rna'])zeros(name,n);
  for(const name of ['motif_pos','ref_atomwise_rasa','ref_pos'])zeros(name,l*3);
  for(const name of ['token_bonds','unindexing_pair_mask'])zeros(name,n*n);
  for(const name of ['atom_to_token_map','ref_space_uid','is_backbone','is_ca','is_central','is_sidechain','is_virtual'])zeros(name,l);
  zeros('ref_atom_name_chars',l*256);zeros('ref_element',l*128);zeros('restype',n*32);
  zeros('ref_motif_token_type',n*3);zeros('terminus_type',n*2);
  zeros('residue_index',n);zeros('token_index',n);zeros('is_protein',n).fill(1);zeros('ref_plddt',n).fill(1);
  const names=['N','CA','C','O','CB',...Array.from({length:9},(_,i)=>'V'+i)];
  for(let i=0;i<n;i++) {
    f.restype[i*32+31]=1;f.ref_motif_token_type[i*3]=1;f.residue_index[i]=f.token_index[i]=i;
    for(let a=0;a<14;a++) {
      const atom=i*14+a;f.atom_to_token_map[atom]=f.ref_space_uid[atom]=i;
      f.is_backbone[atom]=Number(a<4);f.is_ca[atom]=Number(a===1);f.is_central[atom]=Number(a===4);
      f.is_sidechain[atom]=Number(a>=4);f.is_virtual[atom]=Number(a>=5);f.ref_element[atom*128]=1;
      for(let c=0;c<4;c++)f.ref_atom_name_chars[atom*256+c*64+(names[a].padEnd(4)[c].charCodeAt(0)-32)]=1;
    }
  }
  f.terminus_type[1]=1;f.terminus_type[(n-1)*2]=1;
  if(cyclic)f.cyclic_asym_ids=new Float32Array([0]);
  return f;
}

export function validateFeatures(f,config){
  const n=f?.residue_index?.length,l=f?.atom_to_token_map?.length;
  if(!Number.isSafeInteger(n)||n<1||!Number.isSafeInteger(l)||l<1)throw new Error('Features need positive token and atom counts');
  const sizes={};
  for(const [key,width]of Object.entries(config.token_initializer.atom_1d_features))sizes[key]=l*width;
  for(const [key,width]of Object.entries(config.token_initializer.token_1d_features))sizes[key]=n*width;
  for(const key of ['atom_to_token_map','ref_space_uid','is_ca','is_motif_atom_with_fixed_seq','is_motif_atom_with_fixed_coord'])sizes[key]=l;
  for(const key of ['asym_id','entity_id','residue_index','token_index','sym_id','is_motif_token_with_fully_fixed_coord'])sizes[key]=n;
  sizes.motif_pos=l*3;sizes.token_bonds=n*n;sizes.unindexing_pair_mask=n*n;
  for(const [key,size]of Object.entries(sizes)){const values=f[key];if(!values||values.length!==size||!Array.from(values).every(Number.isFinite))throw new Error('Invalid feature '+key+' (expected '+size+' finite values)');}
  // Upstream reserves at least 32 interchain keys when there are >3 chains.
  // Smaller token sets would give a negative intra-chain key count.
  if(n<32&&new Set(f.asym_id).size>3)throw new RangeError('More than three chains requires at least 32 tokens for upstream neighbor selection');
  const members=new Uint32Array(n),representatives=new Uint32Array(n);
  for(let i=0;i<l;i++){const token=f.atom_to_token_map[i];if(!Number.isInteger(token)||token<0||token>=n)throw new Error('Invalid atom-to-token index');members[token]++;if(f.is_ca[i])representatives[token]++;}
  if(members.some(v=>!v)||representatives.some(v=>v!==1))throw new Error('Every token needs atoms and exactly one representative');
  for(const key of ['is_ca','is_motif_atom_with_fixed_seq','is_motif_atom_with_fixed_coord','is_motif_token_with_fully_fixed_coord'])if(Array.from(f[key]).some(v=>v!==0&&v!==1))throw new Error('Invalid boolean mask '+key);
  return {tokens:n,atoms:l};
}
