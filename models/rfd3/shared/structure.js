// Internal row keys are lowercase; output tags use the PDBx dictionary spelling.
const xyz = ['cartn_x', 'cartn_y', 'cartn_z'];
const quoted = Symbol('CIF literal missing-value markers');
const spelling = { group_pdb: 'group_PDB', cartn_x: 'Cartn_x', cartn_y: 'Cartn_y', cartn_z: 'Cartn_z',
  b_iso_or_equiv: 'B_iso_or_equiv', pdbx_pdb_ins_code: 'pdbx_PDB_ins_code',
  pdbx_pdb_model_num: 'pdbx_PDB_model_num' };
const present = value => value != null && value !== '.' && value !== '?';
const has = (row, key) => present(row[key]) || row[quoted]?.has(key);
const field = (row, key, fallback = '') => has(row, key) ? String(row[key]) : fallback;
function number(value, name) {
  const text = String(value ?? '').trim();
  // PDBx permits standard uncertainty, e.g. 1.23(4) or 1.23(4)e2.
  const pattern = /^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\(\d+\))?([eE][+-]?\d+)?$/;
  const m = pattern.exec(text);
  const n = m ? Number(m[1] + (m[2] ?? '')) : NaN;
  if (!Number.isFinite(n)) throw new Error('Invalid numeric field ' + name);
  return n;
}
function checkCoordinates(row) { for (const key of xyz) number(row[key], key); return row; }
function annotatePolymers(rows) {
  const chains = new Map();
  for (const row of rows) {
    if (row.group_pdb !== 'ATOM') continue;
    const chain = field(row, 'label_asym_id', field(row, 'auth_asym_id'));
    const key = JSON.stringify([field(row, 'pdbx_pdb_model_num', '1'), chain]);
    if (!chains.has(key)) chains.set(key, new Map());
    const residues = chains.get(key);
    const residue = JSON.stringify([row.auth_seq_id, row.pdbx_pdb_ins_code]);
    if (!residues.has(residue)) residues.set(residue, residues.size + 1);
    if (!has(row, 'label_seq_id')) row.label_seq_id = String(residues.get(residue));
  }
  return rows;
}
export function parsePDB(source) {
  const rows = [];
  let model = '1';
  for (const line of source.split(/\r?\n/)) {
    const f = (a, b) => line.slice(a, b).trim();
    const group = f(0, 6);
    if (group === 'MODEL') model = f(10, 14);
    if (group !== 'ATOM' && group !== 'HETATM') continue;
    const charge = f(78, 80);
    if (charge && !/^\d[+-]$/.test(charge)) throw new Error('Invalid PDB charge');
    rows.push(checkCoordinates({
      group_pdb: group, id: f(6, 11), type_symbol: f(76, 78) || '?',
      label_atom_id: f(12, 16), auth_atom_id: line.slice(12, 16), label_alt_id: f(16, 17) || '.',
      label_comp_id: f(17, 20), auth_comp_id: f(17, 20), label_asym_id: f(21, 22) || '?',
      auth_asym_id: f(21, 22) || '.', auth_seq_id: f(22, 26), label_entity_id: '?', label_seq_id: '?',
      pdbx_pdb_ins_code: f(26, 27) || '?', cartn_x: f(30, 38), cartn_y: f(38, 46), cartn_z: f(46, 54),
      occupancy: f(54, 60) || '?', b_iso_or_equiv: f(60, 66) || '?', pdbx_pdb_model_num: model,
      pdbx_formal_charge: charge ? String(Number(charge[0]) * (charge[1] === '-' ? -1 : 1)) : '?',
    }));
  }
  if (!rows.length) throw new Error('Missing PDB coordinate records');
  return annotatePolymers(rows);
}
export function parsePDBConect(source) {
  const pairs = [];
  for (const line of source.split(/\r?\n/)) {
    if (line.slice(0, 6).trim() !== 'CONECT') continue;
    const a = line.slice(6, 11).trim();
    if (!a) continue;
    for (let pos = 11; pos < line.length && pos < 31; pos += 5) {
      const b = line.slice(pos, pos + 5).trim();
      if (b) pairs.push([a, b]);
    }
  }
  return pairs;
}
export function parseMMCIFBonds(source) {
  const stream = tokens(source);
  let token = stream.next().value;
  const advance = () => { token = stream.next().value; };
  const byComp = new Map();
  while (token) {
    if (token.quoted || token.value.toLowerCase() !== 'loop_') { advance(); continue; }
    advance();
    const tags = [];
    while (token && !token.quoted && token.value.startsWith('_')) { tags.push(token.value.toLowerCase()); advance(); }
    if (!tags.length) continue;
    const isBond = tags.every(k => k.startsWith('_chem_comp_bond.'));
    const keys = tags.map(k => k.slice(16));
    while (token && (token.quoted || !control.test(token.value))) {
      const row = Object.create(null);
      for (const key of keys) {
        if (!token || (!token.quoted && control.test(token.value))) break;
        if (isBond) row[key] = token.value;
        advance();
      }
      if (isBond && row.comp_id && row.atom_id_1 && row.atom_id_2) {
        if (!byComp.has(row.comp_id)) byComp.set(row.comp_id, []);
        byComp.get(row.comp_id).push([row.atom_id_1, row.atom_id_2]);
      }
    }
  }
  return byComp;
}
export function writePDB(rows) {
  if (!rows.length) throw new Error('Missing coordinate rows');
  const models = new Map(), identities = new Set();
  const fixed = (v, width, left = false) => {
    const text = String(v);
    if (text.length > width || /[^ -~]/.test(text)) throw new Error('Field does not fit PDB: ' + text);
    return left ? text.padEnd(width) : text.padStart(width);
  };
  for (const row of rows) {
    checkCoordinates(row);
    const model = field(row, 'pdbx_pdb_model_num', '1');
    if (!/^\d+$/.test(model) || Number(model) < 1) throw new Error('Invalid PDB model number');
    const author = suffix => field(row, 'auth_' + suffix, field(row, 'label_' + suffix));
    const seq = 'auth_seq_id' in row ? field(row, 'auth_seq_id') : field(row, 'label_seq_id');
    if (!/^[+-]?\d+$/.test(seq) || !Number.isSafeInteger(Number(seq))) throw new Error('PDB auth_seq_id must be an integer');
    const atom = author('atom_id'), element = field(row, 'type_symbol');
    const group = field(row, 'group_pdb', 'ATOM');
    if (!['ATOM', 'HETATM'].includes(group)) throw new Error('Invalid coordinate group');
    if (!atom.trim()) throw new Error('Missing PDB atom name');
    // Preserve an original four-column name, including when element is absent.
    const padded = atom.length === 4 ? atom :
      (atom.length < 4 && !/^\d/.test(atom) && (element.length === 1 || !element) ? ' ' : '') + atom;
    const chain = author('asym_id'), ins = field(row, 'pdbx_pdb_ins_code'), alt = field(row, 'label_alt_id');
    const identity = JSON.stringify([Number(model), chain, Number(seq), ins, atom.trim(), alt]);
    if (identities.has(identity)) throw new Error('PDB atom-identity collision: ' + identity);
    identities.add(identity);
    const decimal = (key, width, places) => fixed(has(row, key) ? number(row[key], key).toFixed(places) : '', width);
    let charge = '';
    if (has(row, 'pdbx_formal_charge')) {
      const n = number(row.pdbx_formal_charge, 'pdbx_formal_charge');
      if (!Number.isInteger(n) || Math.abs(n) > 9) throw new Error('Charge does not fit PDB');
      if (n) charge = Math.abs(n) + (n < 0 ? '-' : '+');
    }
    if (!models.has(model)) models.set(model, []);
    const serial = models.get(model).length + 1;
    const id = field(row, 'id', String(serial));
    if (!/^\d+$/.test(id) || Number(id) < 1) throw new Error('Invalid PDB atom serial');
    models.get(model).push(fixed(group, 6, true) + fixed(id, 5) + ' ' + fixed(padded, 4, true) + fixed(alt, 1) +
      fixed(author('comp_id'), 3) + ' ' + fixed(chain, 1) + fixed(String(Number(seq)), 4) + fixed(ins, 1) + '   ' +
      xyz.map(k => decimal(k, 8, 3)).join('') + decimal('occupancy', 6, 2) + decimal('b_iso_or_equiv', 6, 2) +
      '          ' + fixed(element, 2) + fixed(charge, 2));
  }
  const explicit = models.size > 1 || !models.has('1'), lines = [];
  for (const [model, atoms] of models) {
    if (explicit) lines.push('MODEL     ' + fixed(model, 4));
    lines.push(...atoms);
    if (explicit) lines.push('ENDMDL');
  }
  return lines.concat('END', '').join('\n');
}
function* tokens(source) {
  const lines = source.replaceAll('\r\n', '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i], p = 0;
    if (line.startsWith(';')) {
      const value = [line.slice(1)];
      while (++i < lines.length && !lines[i].startsWith(';')) value.push(lines[i]);
      if (i === lines.length) throw new Error('Unterminated CIF text field');
      yield { value: value.join('\n'), quoted: true };
      line = lines[i]; p = 1;
      if (line.length > 1 && !/\s/.test(line[1])) throw new Error('CIF closing semicolon must be followed by whitespace');
    }
    while (p < line.length) {
      if (/\s/.test(line[p])) { p++; continue; }
      if (line[p] === '#') break;
      const quote = line[p];
      if (quote === "'" || quote === '"') {
        const start = ++p;
        while (p < line.length && !(line[p] === quote && (p + 1 === line.length || /\s/.test(line[p + 1])))) p++;
        if (p === line.length) throw new Error('Unterminated CIF quoted field');
        yield { value: line.slice(start, p++), quoted: true };
      } else {
        const start = p++;
        while (p < line.length && !/\s/.test(line[p])) p++;
        yield { value: line.slice(start, p), quoted: false };
      }
    }
  }
}
const control = /^(?:_|data_|save_|global_$|loop_$|stop_$)/i;
export function parseMMCIF(source) {
  const stream = tokens(source);
  let token = stream.next().value;
  const advance = () => { token = stream.next().value; }, rows = [];
  while (token) {
    if (token.quoted || token.value.toLowerCase() !== 'loop_') { advance(); continue; }
    advance();
    const tags = [];
    while (token && !token.quoted && token.value.startsWith('_')) { tags.push(token.value.toLowerCase()); advance(); }
    if (!tags.length) throw new Error('Empty CIF loop header');
    const atoms = tags.some(k => k.startsWith('_atom_site.'));
    if (atoms && (!tags.every(k => k.startsWith('_atom_site.')) || new Set(tags).size !== tags.length))
      throw new Error('Invalid atom_site header');
    if (atoms && rows.length) throw new Error('Multiple atom_site loops are unsupported');
    const keys = tags.map(k => k.slice(11));
    while (token && (token.quoted || !control.test(token.value))) {
      const row = Object.create(null);
      const literals = new Set();
      for (const key of keys) {
        if (!token || (!token.quoted && control.test(token.value))) throw new Error('Incomplete CIF loop row');
        if (atoms) {
          row[key] = token.value;
          if (token.quoted && (token.value === '?' || token.value === '.')) literals.add(key);
        }
        advance();
      }
      if (literals.size) Object.defineProperty(row, quoted, { value: literals });
      if (atoms) rows.push(checkCoordinates(row));
    }
  }
  if (!rows.length) throw new Error('Missing atom_site coordinate loop');
  return rows;
}
function cifValue(input, literal) {
  const text = String(input ?? '?');
  if (literal && (text === '?' || text === '.')) return "'" + text + "'";
  if (/[\r\n]/.test(text)) {
    if (/(^|\n);/.test(text)) throw new Error('Unsupported CIF multiline delimiter');
    return ';' + text + '\n;';
  }
  if (text && !/\s/.test(text) && !/^['"#;$\[\]]/.test(text) && !control.test(text)) return text;
  if (!text.includes("'")) return "'" + text + "'";
  if (!text.includes('"')) return '"' + text + '"';
  return ';' + text + '\n;';
}
export function writeMMCIF(input) {
  if (!input.length) throw new Error('Missing coordinate rows');
  const rows = annotatePolymers(input.map(row => {
    const copy = { ...row };
    if (row[quoted]) Object.defineProperty(copy, quoted, { value: row[quoted] });
    return copy;
  }));
  // Regenerate identifiers only when missing or duplicated, preserving valid IDs.
  const ids = new Set(rows.map(r => field(r, 'id')));
  if (ids.size !== rows.length || ids.has('')) rows.forEach((r, i) => { r.id = String(i + 1); });
  const keys = new Set();
  for (const row of rows) {
    checkCoordinates(row);
    for (const key of Object.keys(row)) {
      if (!/^[a-z0-9_]+$/.test(key)) throw new Error('Invalid atom_site column name');
      keys.add(key);
    }
  }
  const lines = ['data_structure', '#', 'loop_', ...Array.from(keys, k => '_atom_site.' + (spelling[k] ?? k))];
  for (const row of rows) {
    const values = Array.from(keys, k => cifValue(row[k], row[quoted]?.has(k)));
    lines.push(values.join(values.some(v => v.startsWith(';')) ? '\n' : ' '));
  }
  return lines.concat('#', '').join('\n');
}
