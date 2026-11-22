// Match libstdc++'s introsort partition + insertion finish used by torch.sort.
// Equal uniform logits have a visible ordering when nucleus sampling removes
// part of the vocabulary. A stable JavaScript sort is not equivalent.
export function torchSort21(values) {
  const order = Array.from({ length: 21 }, (_, i) => i),
    less = (a, b) => values[order[a]] > values[order[b]],
    swap = (a, b) => {
      const t = order[a];
      order[a] = order[b];
      order[b] = t;
    },
    stack = [[0, 21]];
  while (stack.length) {
    let [first, last] = stack.pop();
    while (last - first > 16) {
      const a = first + 1,
        b = first + Math.floor((last - first) / 2),
        c = last - 1;
      const median = less(a, b)
        ? less(b, c)
          ? b
          : less(a, c)
            ? c
            : a
        : less(a, c)
          ? a
          : less(b, c)
            ? c
            : b;
      swap(first, median);
      let left = first + 1,
        right = last;
      while (true) {
        while (less(left, first)) left++;
        do {
          right--;
        } while (less(first, right));
        if (left >= right) break;
        swap(left, right);
        left++;
      }
      stack.push([left, last]);
      last = left;
    }
  }
  for (let i = 1; i < 21; i++) {
    const value = order[i];
    let j = i;
    while (j > 0 && values[value] > values[order[j - 1]]) {
      order[j] = order[j - 1];
      j--;
    }
    order[j] = value;
  }
  return order;
}
export const torchSort21WGSL = `
var starts:array<u32,32>;var ends:array<u32,32>;starts[0]=0u;ends[0]=21u;var stackSize=1u;
while(stackSize>0u){stackSize--;let first=starts[stackSize];var last=ends[stackSize];
while(last-first>16u){let a=first+1u;let b=first+(last-first)/2u;let c=last-1u;var median=b;
if(probs[order[a]]>probs[order[b]]){if(probs[order[b]]>probs[order[c]]){median=b;}else if(probs[order[a]]>probs[order[c]]){median=c;}else{median=a;}}
else{if(probs[order[a]]>probs[order[c]]){median=a;}else if(probs[order[b]]>probs[order[c]]){median=c;}}
let pivot=order[first];order[first]=order[median];order[median]=pivot;
var left=first+1u;var right=last;loop{while(probs[order[left]]>probs[order[first]]){left++;}right--;while(probs[order[first]]>probs[order[right]]){right--;}if(left>=right){break;}let t=order[left];order[left]=order[right];order[right]=t;left++;}
starts[stackSize]=left;ends[stackSize]=last;stackSize++;last=left;}}
for(var j=1u;j<21u;j++){let value=order[j];var k=j;while(k>0u){if(!(probs[value]>probs[order[k-1u]])){break;}order[k]=order[k-1u];k--;}order[k]=value;}
`;
