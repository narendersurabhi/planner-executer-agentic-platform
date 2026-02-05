(function(){
  'use strict';

  // Utilities
  function abs(n){ return Math.abs(n); }
  function gcd(a,b){ a=abs(a); b=abs(b); while(b){ const t=b; b=a%b; a=t; } return a||1; }
  function lcm(a,b){ return abs(a*b)/gcd(a,b); }
  function simplify(n,d){ const g=gcd(n,d); n/=g; d/=g; if(d<0){ n=-n; d=-d; } return {n,d}; }

  function toMixed(n,d){ const s=simplify(n,d); n=s.n; d=s.d; const sign = n<0?-1:1; const whole = sign*Math.floor(abs(n)/d); const rem = abs(n)%d; return {whole, n: sign*rem, d}; }

  function formatFraction(n,d){ const s=simplify(n,d); return `${s.n}/${s.d}`; }
  function formatMixed(n,d){ const m=toMixed(n,d); if(m.n===0) return `${m.whole}`; if(m.whole===0) return `${m.n}/${m.d}`; return `${m.whole} ${abs(m.n)}/${m.d}`; }

  // Operations with steps
  function addFractions(a,b,c,d){
    const L = lcm(b,d);
    const a2 = a*(L/b);
    const c2 = c*(L/d);
    const sum = a2 + c2;
    const simp = simplify(sum,L);
    const steps = [
      `Find a common bottom number (denominator). The LCM of ${b} and ${d} is ${L}.`,
      `Make the tops match the new bottom: ${a} becomes ${a2} (because ${a}×${L/b}=${a2}), and ${c} becomes ${c2} (because ${c}×${L/d}=${c2}).`,
      `Add the tops and keep the bottom: ${a2}+${c2}=${sum} over ${L}.`,
      `Simplify the fraction to make it neat: ${sum}/${L} = ${simp.n}/${simp.d}.`
    ];
    return {n:simp.n, d:simp.d, steps};
  }

  function mulFractions(a,b,c,d){
    const n = a*c;
    const den = b*d;
    const simp = simplify(n,den);
    const steps = [
      `Multiply the tops: ${a}×${c}=${n}.`,
      `Multiply the bottoms: ${b}×${d}=${den}.`,
      `Simplify to make it neat: ${n}/${den} = ${simp.n}/${simp.d}.`
    ];
    return {n:simp.n, d:simp.d, steps};
  }

  function divFractions(a,b,c,d){
    // a/b ÷ c/d = a/b × d/c
    const rn = d; const rd = c;
    const n = a*rn;
    const den = b*rd;
    const simp = simplify(n,den);
    const steps = [
      `Dividing by a fraction is the same as multiplying by its flip (reciprocal). Flip ${c}/${d} to get ${rn}/${rd}.`,
      `Now multiply: ${a}/${b} × ${rn}/${rd}.`,
      `Multiply the tops: ${a}×${rn}=${n}.`,
      `Multiply the bottoms: ${b}×${rd}=${den}.`,
      `Simplify to make it neat: ${n}/${den} = ${simp.n}/${simp.d}.`
    ];
    return {n:simp.n, d:simp.d, steps};
  }

  function compute(a,b,op,c,d){
    if(op==='add') return addFractions(a,b,c,d);
    if(op==='mul') return mulFractions(a,b,c,d);
    if(op==='div') return divFractions(a,b,c,d);
    throw new Error('Unknown operation');
  }

  // Similar question generation
  function clamp(n,min,max){ return Math.max(min, Math.min(max,n)); }
  function vary(v){
    const delta = [ -1, 1, 2 ][Math.floor(Math.random()*3)];
    return Math.max(1, v + delta);
  }
  function generateSimilar(a,b,op,c,d,count=3){
    const out=[];
    let attempts=0;
    while(out.length<count && attempts<20){
      attempts++;
      const a2 = clamp(vary(a),1,20);
      const b2 = clamp(vary(b),1,20);
      const c2 = clamp(vary(c),1,20);
      const d2 = clamp(vary(d),1,20);
      try{
        const r = compute(a2,b2,op,c2,d2);
        out.push({ a:a2,b:b2,c:c2,d:d2, op, ansFrac: formatFraction(r.n,r.d), ansMixed: formatMixed(r.n,r.d) });
      }catch(e){ /* skip */ }
    }
    return out;
  }

  // DOM wiring
  function $(id){ return document.getElementById(id); }
  function clearEl(el){ while(el.firstChild) el.removeChild(el.firstChild); }

  function showResult(n,d){
    $('resultFraction').textContent = formatFraction(n,d);
    $('resultMixed').textContent = `(mixed number) ${formatMixed(n,d)}`;
  }

  function showSteps(steps){
    const list = $('steps');
    clearEl(list);
    steps.forEach(s=>{ const li=document.createElement('li'); li.textContent=s; list.appendChild(li); });
  }

  function showSimilar(sim,op){
    const list = $('similarList');
    clearEl(list);
    const symbol = op==='add'?'+':(op==='mul'?'×':'÷');
    sim.forEach(item=>{
      const li=document.createElement('li');
      const q = `${item.a}/${item.b} ${symbol} ${item.c}/${item.d}`;
      li.innerHTML = `<div>Q: <strong>${q}</strong></div><div>A: ${item.ansFrac} (mixed: ${item.ansMixed})</div>`;
      list.appendChild(li);
    });
  }

  function validateInputs(a,b,c,d){
    const ints=[a,b,c,d].every(Number.isInteger);
    if(!ints) throw new Error('Please use whole numbers only.');
    if(b===0 || d===0) throw new Error('Denominators cannot be 0.');
    if(a<1||b<1||c<1||d<1) throw new Error('Use positive whole numbers (1 or more).');
  }

  function onCalculate(){
    const a = parseInt($('num1').value,10);
    const b = parseInt($('den1').value,10);
    const c = parseInt($('num2').value,10);
    const d = parseInt($('den2').value,10);
    const op = $('opSelect').value;
    try{
      validateInputs(a,b,c,d);
      const res = compute(a,b,op,c,d);
      showResult(res.n,res.d);
      showSteps(res.steps);
      const sim = generateSimilar(a,b,op,c,d,3);
      showSimilar(sim,op);
    }catch(err){
      showResult(0,1);
      showSteps([err.message]);
      showSimilar([],op);
    }
  }

  function init(){
    $('calculateBtn').addEventListener('click', onCalculate);
    onCalculate(); // compute once with defaults
  }

  if(document.readyState==='loading'){
    document.addEventListener('DOMContentLoaded', init);
  }else{
    init();
  }
})();
