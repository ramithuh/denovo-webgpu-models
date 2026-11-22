// PyTorch 2.5 CPU topk: libstdc++ introselect, then introsort [0,k-1).
// The tie permutation matters for padded residues. L<=2048, K=min(48,L),
// so PyTorch's alternative k*64<=L partial-sort path is never selected.
export function topk(values, k) {
  const q = Array.from(values, (_, i) => i),
    less = (a, b) => values[a] < values[b];
  const swap = (a, b) => {
    const t = q[a];
    q[a] = q[b];
    q[b] = t;
  };
  function median(out, a, b, c) {
    if (less(q[a], q[b])) {
      if (less(q[b], q[c])) swap(out, b);
      else if (less(q[a], q[c])) swap(out, c);
      else swap(out, a);
    } else if (less(q[a], q[c])) swap(out, a);
    else if (less(q[b], q[c])) swap(out, c);
    else swap(out, b);
  }
  function partition(lo, hi) {
    median(lo, lo + 1, lo + Math.floor((hi - lo) / 2), hi - 1);
    const pivot = q[lo];
    let a = lo + 1,
      b = hi;
    while (true) {
      while (less(q[a], pivot)) a++;
      b--;
      while (less(pivot, q[b])) b--;
      if (a >= b) return a;
      swap(a++, b);
    }
  }
  function insertion(lo, hi) {
    for (let i = lo + 1; i < hi; i++) {
      const value = q[i];
      let j = i;
      while (j > lo && less(value, q[j - 1])) {
        q[j] = q[j - 1];
        j--;
      }
      q[j] = value;
    }
  }
  function adjust(lo, hole, len, value) {
    const top = hole;
    let child = hole;
    while (child < Math.floor((len - 1) / 2)) {
      child = 2 * (child + 1);
      if (less(q[lo + child], q[lo + child - 1])) child--;
      q[lo + hole] = q[lo + child];
      hole = child;
    }
    if (len % 2 === 0 && child === (len - 2) / 2) {
      child = 2 * (child + 1);
      q[lo + hole] = q[lo + child - 1];
      hole = child - 1;
    }
    let parent = (hole - 1) >> 1;
    while (hole > top && less(q[lo + parent], value)) {
      q[lo + hole] = q[lo + parent];
      hole = parent;
      parent = (hole - 1) >> 1;
    }
    q[lo + hole] = value;
  }
  function makeHeap(lo, hi) {
    const n = hi - lo;
    for (let p = (n - 2) >> 1; p >= 0; p--) adjust(lo, p, n, q[lo + p]);
  }
  function heapSelect(lo, mid, hi) {
    makeHeap(lo, mid);
    for (let i = mid; i < hi; i++)
      if (less(q[i], q[lo])) {
        const v = q[i];
        q[i] = q[lo];
        adjust(lo, 0, mid - lo, v);
      }
  }
  let lo = 0,
    hi = q.length,
    depth = 2 * Math.floor(Math.log2(q.length));
  while (hi - lo > 3) {
    if (depth === 0) {
      heapSelect(lo, k, hi);
      swap(lo, k - 1);
      break;
    }
    depth--;
    const cut = partition(lo, hi);
    if (cut <= k - 1) lo = cut;
    else hi = cut;
  }
  if (hi - lo <= 3) insertion(lo, hi);
  function intro(a, b, d) {
    while (b - a > 16) {
      if (d === 0) {
        makeHeap(a, b);
        while (b - a > 1) {
          const v = q[b - 1];
          q[b - 1] = q[a];
          adjust(a, 0, --b - a, v);
        }
        return;
      }
      const cut = partition(a, b);
      intro(cut, b, --d);
      b = cut;
    }
  }
  if (k > 1) {
    intro(0, k - 1, 2 * Math.floor(Math.log2(k - 1)));
    insertion(0, k - 1);
  }
  return Int32Array.from(q.slice(0, k));
}
