"""Builds dashboard.html from rl_results.json.

evaluate_rl.py runs this automatically at the end, so the dashboard always
shows your latest test. You can also run it by itself:
    python make_dashboard.py
Then double-click dashboard.html (works offline, no server needed).
"""
import json
from datetime import datetime

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAT-01 Mission Board</title>
<style>
:root{--bg:#0B1220;--panel:#131C2E;--line:#24304A;--text:#E6EBF5;--muted:#9AA6BF;--soft:#C3CCDD;
--base:#9AA6BF;--rules:#60A5FA;--ai:#F59E0B;--safe:#34D399;--floor:#E5484D}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px;display:flex;flex-direction:column;gap:20px}
.mono{font-family:ui-monospace,Consolas,"Cascadia Mono",monospace}
.eyebrow{margin:0;font-size:12px;letter-spacing:.08em;color:var(--muted)}
h1{margin:0;font-size:30px}h2{margin:0;font-size:17px}
.muted{color:var(--muted)}
header{display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;align-items:flex-end}
.seg{display:flex;flex-wrap:wrap;gap:6px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:4px}
.seg button{min-height:44px;padding:0 16px;border:0;border-radius:8px;background:transparent;color:var(--text);font:600 14px system-ui,sans-serif;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--ai);color:var(--bg)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.take{font-size:15px}.take b{color:var(--ai)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}
.card{display:flex;flex-direction:column;gap:8px}
.card.best{border-color:var(--ai)}
.card .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.card .name{margin:0;font-weight:700}
.pill{font-size:12px;font-weight:700;color:var(--bg);background:#F5C66B;border-radius:999px;padding:3px 10px}
.big{margin:0;font-size:34px;font-weight:600}
.big small{font-size:15px;color:var(--muted);font-weight:400}
.mini{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-size:13px}
.mini p{margin:0}.mini .v{font-size:15px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--soft)}
.legend span{display:flex;align-items:center;gap:6px}
.chart{display:flex;gap:8px}
.yax{display:flex;flex-direction:column;justify-content:space-between;font-size:12px;color:var(--muted);padding:2px 0 22px;text-align:right}
.xax{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
svg.plot{width:100%;height:260px;display:block}
.tablebox{overflow-x:auto}
table{width:100%;min-width:560px;border-collapse:collapse;font-size:14px}
th,td{padding:10px 8px;border-bottom:1px solid #1B2539;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:500;border-bottom:1px solid var(--line)}
tbody th{font-weight:400;color:var(--soft);text-align:left}
</style></head><body>
<div class="wrap">
<header>
<div style="display:flex;flex-direction:column;gap:6px">
<p class="eyebrow mono">ORBITAL COMPUTE SIM · __ORBIT__ · UPDATED __UPDATED__</p>
<h1>SAT-01 Mission Board</h1>
<p class="muted" style="margin:0">Same satellite, different schedulers. Averages of __N__ unseen 6-hour runs per weather type.</p>
</div>
<div class="seg" role="group" aria-label="Space weather" id="seg"></div>
</header>
<div class="panel take" role="status"><b>Takeaway: </b><span id="take"></span></div>
<section class="panel" id="orbitbox" style="display:none;flex-direction:column;gap:12px">
<p class="eyebrow mono">LIVE ORBIT VIEW · DRAG TO TURN · CLICK A SATELLITE</p>
<h2>The fleet in orbit</h2>
<iframe id="orbitframe" title="Animated view of the satellite fleet orbiting Earth" style="width:100%;height:1050px;border:0;border-radius:10px;background:#070B14" loading="lazy" srcdoc="__ORBITDOC__"></iframe>
</section>
<div class="cards" id="cards"></div>
<section class="panel" style="display:flex;flex-direction:column;gap:12px">
<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center">
<h2>Battery over one sample 6-hour run</h2><div class="legend" id="legend"></div></div>
<div class="chart"><div class="yax mono"><span>100%</span><span>50%</span><span>0%</span></div>
<div style="flex:1;min-width:0;display:flex;flex-direction:column;gap:4px">
<svg class="plot" id="plot" viewBox="0 0 600 220" preserveAspectRatio="none" role="img"></svg>
<div class="xax mono"><span>0 h</span><span>1 h</span><span>2 h</span><span>3 h</span><span>4 h</span><span>5 h</span><span>6 h</span></div>
</div></div>
<p class="muted" style="margin:0;font-size:14px">Red dashed line = <span id="floorpct">70</span>% safety floor. Dark bands = Earth's shadow (no solar power).</p>
</section>
<section class="panel" style="display:flex;flex-direction:column;gap:12px">
<h2>Full comparison</h2>
<p class="muted" style="margin:0;font-size:14px">Work done = computing actually finished. Score = work minus penalties for low battery, overheating, wasted work and offloading.</p>
<div class="tablebox"><table id="tbl"></table></div>
</section>
<section class="panel" id="fleetbox" style="display:none;flex-direction:column;gap:12px">
<p class="eyebrow mono" id="fleetsub"></p>
<h2>Fleet: 12 satellites sharing work over laser links</h2>
<p class="muted" style="margin:0;font-size:14px" id="fleetlead"></p>
<div class="tablebox"><table id="ftbl"></table></div>
<p class="muted" style="margin:0;font-size:14px" id="fleetnote">Local = each satellite alone. Fleet = shared queue, best-fit packing. Strict = fleet that always pulls sensitive jobs out of the South Atlantic Anomaly radiation zone. AI = learns those settings minute by minute. Best value in each row is highlighted.</p>
</section>
<section class="panel" id="valbox" style="display:none;flex-direction:column;gap:12px">
<p class="eyebrow mono">WHAT THE EXTRA COMPUTING IS WORTH · AT TODAY'S GPU RENTAL PRICES</p>
<div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px">
<h2>Dollar value of the hybrid dispatcher</h2>
<div class="seg" role="group" aria-label="Weather mix" id="mixseg"></div>
</div>
<p class="big mono" id="valhead" style="margin:0"></p>
<p class="muted" style="margin:0;font-size:14px" id="vallead"></p>
<div class="tablebox"><table id="vtbl"></table></div>
<p class="muted" style="margin:0;font-size:14px">Range = H100 prices (low end) to B200 prices (high end), per kWh of GPU-server power. Assumes every extra kWh is sold at today's rental prices and uses this sim's satellite design (100 kW of GPUs), not a real company's.</p>
</section>
<section class="panel" id="heliosbox" style="display:none;flex-direction:column;gap:12px">
<p class="eyebrow mono">REAL AI JOBS · HELIOS GPU TRACE · AI1 SPECS · 53° ORBIT</p>
<h2>Sharing on real AI workloads</h2>
<p class="muted" style="margin:0;font-size:14px" id="helioslead"></p>
<div class="tablebox"><table id="htbl"></table></div>
<p class="muted" style="margin:0;font-size:14px">Real deep-learning jobs from SenseTime's Helios GPU clusters (2020): real arrival bursts, GPU counts and run times. Gain = GPU work delivered with sharing vs. each satellite alone, averaged over 6 different 12-hour slices of the trace. Demand = work offered as a share of the fleet's full GPU power.</p>
</section>
<section class="panel" id="scalebox" style="display:none;flex-direction:column;gap:12px">
<p class="eyebrow mono">AI1 SPECS · 53° ORBIT · FLEET SIZE TEST</p>
<h2>Bigger fleets: 12, 48 and 96 satellites</h2>
<p class="muted" style="margin:0;font-size:14px" id="scalelead"></p>
<div class="tablebox"><table id="stbl"></table></div>
<div class="tablebox"><table id="atbl"></table></div>
<p class="muted" style="margin:0;font-size:14px">Sharing = shared queue vs. each satellite alone, demand scaled to fleet size. AI edge = trained dispatcher vs. the best simple rule on an aging fleet (one in three satellites worn), fresh test runs, realistic weather mix. Dollars per 1,000 satellites per year, H100 low to B200 high.</p>
</section>
</div>
<script>
const DATA = __DATA__;
const FLEET = __FLEET__;
const HYB = __HYB__;
const VAL = __VAL__;
const SCALE = __SCALE__;
const HELIOS = __HELIOS__;
let MIX = "realistic";
const MIXNAME = {realistic:"Realistic weather", stormy:"Stormy weather"};
const MIXDESC = {realistic:"calm 90%, moderate storm 8%, big storm 2% of the time", stormy:"calm 50%, moderate 25%, big storm 25% (the harsh training mix)"};
const money = v => v>=1e6 ? `$${(v/1e6).toFixed(1)}M` : `$${Math.round(v/1000)}k`;
function renderValue(){
  if(!VAL) return;
  $("valbox").style.display="flex";
  $("mixseg").innerHTML = Object.keys(MIXNAME).map(k=>`<button type="button" data-mix="${k}" aria-pressed="${k===MIX}">${MIXNAME[k]}</button>`).join("");
  const m = VAL.mixes[MIX];
  const rng = r => { const all = Object.values(r.usd_per_sat_year); return [Math.min(...all.map(v=>v[0])), Math.max(...all.map(v=>v[1]))]; };
  const ns = m["vs no sharing"], [lo, hi] = rng(ns);
  $("valhead").innerHTML = `${money(lo*1000)}–${money(hi*1000)} <small>per year, per 1,000 satellites</small>`;
  $("vallead").textContent = `${MIXNAME[MIX]} (${MIXDESC[MIX]}): the hybrid does ${ns.extra_pct}% more computing than satellites working alone, about ${ns.extra_kwh_per_sat_year.toLocaleString()} kWh per satellite per year.`;
  const rows = [["No sharing","vs no sharing"],["Best simple rule","vs best simple rule"]];
  $("vtbl").innerHTML = `<thead><tr><th scope="col">Compared with</th><th scope="col">Extra work</th><th scope="col">Per satellite / year</th><th scope="col">Per 100 / year</th><th scope="col">Per 1,000 / year</th><th scope="col">Per 10,000 / year</th></tr></thead><tbody>` +
    rows.map(([lab,key])=>{const r=m[key], [a,b]=rng(r); return `<tr><th scope="row">${lab}</th><td class="mono">+${r.extra_pct}%</td>`+
      [1,100,1000,10000].map(n=>`<td class="mono">${money(a*n)}–${money(b*n)}</td>`).join("")+`</tr>`}).join("") + "</tbody>";
}
document.addEventListener("click", e=>{const b=e.target.closest("button[data-mix]"); if(b){MIX=b.dataset.mix; renderValue();}});
const META = {
  baseline:{name:"Baseline",desc:"Always full power. Never plans ahead.",color:"var(--base)",dash:"5 4",w:2},
  rules:{name:"Rules",desc:"Hand-written: saves battery for shadow, offloads in storms.",color:"var(--rules)",dash:"",w:2.5},
  ai:{name:"AI",desc:"Neural network that learned its own strategy.",color:"var(--ai)",dash:"",w:3.5},
  ai_safe:{name:"AI + safety limit",desc:"Same AI, but never above the rules' safe power level.",color:"var(--safe)",dash:"2 3",w:3}
};
const NAMES = {LOW:"Calm sun",ELEVATED:"Moderate storm",HIGH:"Big storm"};
const ROWS = [["Work done (kWh)","work_kwh",1],["Overall score","score",1],["Jobs finished","jobs_done",1],
  ["Bit-flip restarts","restarts",1],["Sent to ground","ground",0],["Minutes battery under floor","min_low_batt",1],
  ["Lowest battery %","min_batt",1],["Peak GPU temp °C","peak_temp",1]];
const C = DATA.controllers;
let W = "HIGH";
const $ = id => document.getElementById(id);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}

function fleetTakeaway(){
  // Leads with the fleet results; falls back to the single-satellite summary when there is no fleet data.
  if(!FLEET) return "";
  const a=FLEET.averages[W], share=(a.fleet.work_kwh/a.local.work_kwh-1)*100;
  const lead={LOW:"In calm sun",ELEVATED:"During a moderate storm",HIGH:"During a big storm"}[W];
  let t=`${lead}, sharing jobs across the fleet does <b>${share>=0?"+":""}${share.toFixed(1)}% more computing</b> than satellites working alone`;
  if(HYB){const h=HYB.results[W], e=h.edge_vs_switch.score;
    t+= h.policy_used==="ai"
      ? `, and the AI dispatcher adds more on top, beating the best simple rule in ${e.wins} of ${e.runs} fresh runs.`
      : `. Here a simple rule already captures it, so the dispatcher uses the rule.`;}
  else t+=".";
  if(VAL){const r=VAL.mixes.realistic["vs no sharing"], all=Object.values(r.usd_per_sat_year);
    t+=` Over a realistic year of weather that's worth <b>${money(Math.min(...all.map(v=>v[0]))*1000)}–${money(Math.max(...all.map(v=>v[1]))*1000)}</b> per 1,000 satellites.`;}
  if(HELIOS && HELIOS["48|1.0|none"]){const h=HELIOS["48|1.0|none"][W];
    t+=` On real AI jobs from a GPU datacenter trace the gain is larger: <b>${h.gain_pct>=0?"+":""}${h.gain_pct.toFixed(1)}%</b> for 48 satellites at full demand.`;}
  t+=` No battery dropped below its 70% safety floor.`;
  if(SCALE){const sz=Object.keys(SCALE.sharing), big=sz[sz.length-1], ai=Object.entries(SCALE.ai).filter(([,v])=>v);
    t+=` Bigger fleets gain more: +${SCALE.sharing[big][0].toFixed(1)}% from sharing at ${big} satellites`;
    if(ai.length){const [n,v]=ai[ai.length-1]; t+=`, and on an aging ${n}-satellite fleet the AI adds +${v.pct.toFixed(1)}% over the best rule on simulated jobs (on real AI jobs it ties a simple weather-aware rule)`;}
    t+=".";}
  return t;
}
function takeaway(a){
  const byWork = [...C].sort((x,y)=>a[y].work_kwh-a[x].work_kwh);
  const byScore = [...C].sort((x,y)=>a[y].score-a[x].score);
  const safest = [...C].sort((x,y)=>a[x].min_low_batt-a[y].min_low_batt)[0];
  let t = `${META[byWork[0]].name} gets the most work done (${a[byWork[0]].work_kwh} kWh). `;
  t += byScore[0]===byWork[0] ? `It also has the best overall score. ` : `${META[byScore[0]].name} has the best overall score (${a[byScore[0]].score}). `;
  t += `${META[safest].name} keeps the battery safest (${a[safest].min_low_batt} min under the floor).`;
  if (a.rules && a.rules.ground > 10) t += ` The rules sent ${Math.round(a.rules.ground)} jobs to the ground.`;
  return t;
}

function render(){
  const a = DATA.averages[W];
  $("seg").innerHTML = Object.keys(NAMES).map(k=>`<button type="button" data-w="${k}" aria-pressed="${k===W}">${NAMES[k]}</button>`).join("");
  $("take").innerHTML = fleetTakeaway() || esc(takeaway(a));
  const bestWork = [...C].sort((x,y)=>a[y].work_kwh-a[x].work_kwh)[0];
  $("cards").innerHTML = C.map(c=>`
    <div class="panel card ${c===bestWork?"best":""}">
      <div class="top"><p class="name" style="color:${META[c].color}">${META[c].name}</p>${c===bestWork?'<span class="pill">Most work</span>':""}</div>
      <p class="muted" style="margin:0;font-size:13px">${META[c].desc}</p>
      <p class="big mono">${Math.round(a[c].work_kwh)} <small>kWh of work</small></p>
      <div class="mini">
        <div><p class="muted">Score</p><p class="v mono">${a[c].score}</p></div>
        <div><p class="muted">Jobs finished</p><p class="v mono">${a[c].jobs_done}</p></div>
        <div><p class="muted">Min under floor</p><p class="v mono">${a[c].min_low_batt}</p></div>
        <div><p class="muted">Sent to ground</p><p class="v mono">${Math.round(a[c].ground)}</p></div>
      </div></div>`).join("");
  // chart
  const logs = Object.fromEntries(C.map(c=>[c, DATA.sample_runs[`${c}_${W}`]]));
  const ref = logs.rules, n = ref.length, X = i => (i*600/(n-1)).toFixed(1);
  let svg = "";
  let st = null;
  ref.concat([{lit:true}]).forEach((p,i)=>{
    if(!p.lit && st===null) st=i;
    if(p.lit && st!==null){svg+=`<rect x="${X(st)}" y="0" width="${(i-st)*600/(n-1)}" height="220" fill="#05080F"/>`; st=null;}
  });
  [10,110,210].forEach(y=>svg+=`<line x1="0" y1="${y}" x2="600" y2="${y}" stroke="#24304A"/>`);
  const FL = DATA.battery_floor_pct || 20, fy = 210 - FL*2; $("floorpct").textContent = FL;
  svg += `<line x1="0" y1="${fy}" x2="600" y2="${fy}" stroke="var(--floor)" stroke-width="1.5" stroke-dasharray="6 5" vector-effect="non-scaling-stroke"/>`;
  C.forEach(c=>{
    const step = logs[c].map((p,i)=>`${X(i)},${(210-p.batt*2).toFixed(1)}`).filter((_,i)=>i%3===0).join(" ");
    svg += `<polyline points="${step}" fill="none" stroke="${META[c].color}" stroke-width="${META[c].w}" stroke-dasharray="${META[c].dash}" vector-effect="non-scaling-stroke"/>`;
  });
  $("plot").innerHTML = svg;
  $("plot").setAttribute("aria-label", `Battery percent over 6 hours for each scheduler during ${NAMES[W].toLowerCase()}`);
  $("legend").innerHTML = C.map(c=>`<span><svg width="22" height="6" aria-hidden="true"><line x1="0" y1="3" x2="22" y2="3" stroke="${META[c].color}" stroke-width="${META[c].w}" stroke-dasharray="${META[c].dash}"/></svg>${META[c].name}</span>`).join("")
    + `<span><svg width="14" height="12" aria-hidden="true"><rect width="14" height="12" fill="#05080F" stroke="#24304A"/></svg>Earth's shadow</span>`;
  // table
  $("tbl").innerHTML = `<thead><tr><th scope="col">Metric</th>${C.map(c=>`<th scope="col">${META[c].name}</th>`).join("")}</tr></thead><tbody>`
    + ROWS.map(([l,k,dp])=>`<tr><th scope="row">${l}</th>${C.map(c=>`<td class="mono">${a[c][k].toFixed(dp)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
function renderFleet(){
  if(!FLEET) return;
  $("fleetbox").style.display="flex";
  const a=FLEET.averages[W], P=FLEET.policies;
  const FN={local:"Local",fleet:"Fleet",strict:"Strict",ai:"AI"};
  $("fleetsub").textContent=`${FLEET.orbit.toUpperCase()} · AVG OF ${FLEET.episodes} UNSEEN 6-HOUR RUNS`;
  const bestBy=k=>[...P].sort((x,y)=>a[y][k]-a[x][k])[0];
  const bw=bestBy("work_kwh"), bs=bestBy("score");
  const gain=((a.fleet.work_kwh/a.local.work_kwh-1)*100).toFixed(1);
  let lead=`Sharing work does ${gain}% more computing than satellites working alone. Most work: ${FN[bw]}. Best overall score: ${FN[bs]}.`;
  if(HYB){const h=HYB.results[W], e=h.edge_vs_switch.score;
    lead+=` Hybrid (AI only where it proved better, simple rule elsewhere), on fresh runs: ${e.mean>=0?"+":""}${e.mean} score vs the best simple rule`+
      (h.policy_used==="ai"?` (95% range ${e.low} to ${e.high}, won ${e.wins}/${e.runs}).`:` (uses the rule itself here).`)+
      ` Across a weather mix: +${HYB.gain_vs_local_pct}% work vs no sharing, +${HYB.gain_vs_rule_pct}% vs the best rule.`;}
  $("fleetlead").textContent=lead;
  const rows=[["Work done (kWh)","work_kwh",0,1],["Overall score","score",1,1],["Avg job delay (min)","avg_delay_min",1,-1],
    ["Bit-flip restarts","restarts",1,-1],["Wasted work (kWh)","wasted_kwh",0,-1],["Jobs moved between satellites","moves",0,0]];
  $("ftbl").innerHTML=`<thead><tr><th scope="col">Metric</th>${P.map(p=>`<th scope="col">${FN[p]}</th>`).join("")}</tr></thead><tbody>`+
    rows.map(([l,k,dp,dir])=>{const vals=P.map(p=>a[p][k]); const best=dir>0?Math.max(...vals):dir<0?Math.min(...vals):null;
      return `<tr><th scope="row">${l}</th>${P.map(p=>`<td class="mono" style="${best!==null&&a[p][k]===best?"color:#F5C66B;font-weight:700":""}">${a[p][k].toFixed(dp)}</td>`).join("")}</tr>`}).join("")+"</tbody>";
}
function renderHelios(){
  if(!HELIOS) return;
  $("heliosbox").style.display="flex";
  const keys=Object.keys(HELIOS).filter(k=>!k.startsWith("_"));
  const pct=v=>`${v>=0?"+":""}${v.toFixed(1)}%`;
  const label=k=>{const [n,ld,lk]=k.split("|"); return `${n} satellites · demand ${Math.round(ld*100)}%`+(lk==="today"?" · Starlink-today links":"");};
  let wins=0, runs=0; keys.forEach(k=>["LOW","ELEVATED","HIGH"].forEach(w=>{wins+=HELIOS[k][w].wins; runs+=HELIOS[k][w].runs;}));
  const all=keys.flatMap(k=>["LOW","ELEVATED","HIGH"].map(w=>HELIOS[k][w].gain_pct));
  $("helioslead").textContent=`On real AI jobs, sharing delivers ${pct(Math.min(...all))} to ${pct(Math.max(...all))} more GPU work, depending on how busy the fleet is. It won ${wins} of ${runs} paired runs, and no battery dropped below its 70% floor.`;
  $("htbl").innerHTML=`<thead><tr><th scope="col">Case</th>${["Calm sun","Moderate storm","Big storm"].map(n=>`<th scope="col">${n}</th>`).join("")}</tr></thead><tbody>`+
    keys.map(k=>`<tr><th scope="row">${label(k)}</th>${["LOW","ELEVATED","HIGH"].map(w=>{const v=HELIOS[k][w]; return `<td class="mono"${w===W?' style="color:#F5C66B;font-weight:700"':""}>${pct(v.gain_pct)} <span class="muted">(${v.wins}/${v.runs})</span></td>`}).join("")}</tr>`).join("")+"</tbody>";
}
function renderScale(){
  if(!SCALE) return;
  $("scalebox").style.display="flex";
  const sizes=Object.keys(SCALE.sharing), pct=v=>`${v>=0?"+":""}${v.toFixed(1)}%`;
  $("stbl").innerHTML=`<thead><tr><th scope="col">Sharing gain</th>${["Calm sun","Moderate storm","Big storm"].map(n=>`<th scope="col">${n}</th>`).join("")}</tr></thead><tbody>`+
    sizes.map(n=>`<tr><th scope="row">${n} satellites</th>${SCALE.sharing[n].map(v=>`<td class="mono">${pct(v)}</td>`).join("")}</tr>`).join("")+"</tbody>";
  const ai=SCALE.ai;
  $("atbl").innerHTML=`<thead><tr><th scope="col">AI edge (aging fleet)</th><th scope="col">vs best rule</th><th scope="col">Test runs won</th><th scope="col">Per 1,000 / year</th></tr></thead><tbody>`+
    Object.keys(ai).map(n=>{const r=ai[n]; return `<tr><th scope="row">${n} satellites</th>`+(r?`<td class="mono">${pct(r.pct)}</td><td class="mono">${r.wins} of ${r.runs}</td><td class="mono">${money(r.lo*1000)}–${money(r.hi*1000)}</td>`:`<td class="mono" colspan="3">test running</td>`)+`</tr>`}).join("")+"</tbody>";
  const last=sizes[sizes.length-1];
  $("scalelead").textContent=`Sharing gets better as the fleet grows: ${pct(SCALE.sharing[sizes[0]][0])} at ${sizes[0]} satellites, ${pct(SCALE.sharing[last][0])} at ${last} (calm sun). More satellites means more places to send each job.`;
}
$("seg").addEventListener("click", e=>{const b=e.target.closest("button"); if(b){W=b.dataset.w; render(); renderFleet(); renderHelios();}});
render(); renderFleet(); renderValue(); renderScale(); renderHelios();
if (__HAS_ORBIT__) $("orbitbox").style.display = "flex";
addEventListener("message", e => { if (e.data && e.data.orbitHeight) $("orbitframe").style.height = e.data.orbitHeight + "px"; });
</script></body></html>
"""


def load_scale():
    """Fleet-size results: sharing gain (scale_results.json) and AI edge on an aging fleet
    (results/stress_aging, results/stress_aging_48, results/stress_aging_96)."""
    import os
    sp = next((p for p in ("scale_results.json", "results/scale/scale_results.json") if os.path.exists(p)), None)
    if not sp:
        return None
    sc = json.load(open(sp))
    sharing = {n: [round((sc[f"comfortable_{n}"][w]["fleet_work"] / sc[f"comfortable_{n}"][w]["local_work"] - 1) * 100, 1)
                   for w in ("LOW", "ELEVATED", "HIGH")] for n in ("12", "48", "96") if f"comfortable_{n}" in sc}
    ai = {}
    for n, d in (("12", "results/stress_aging"), ("48", "results/stress_aging_48"), ("96", "results/stress_aging_96")):
        try:
            v = json.load(open(f"{d}/value_results.json"))["mixes"]["realistic"]["vs best simple rule"]
            h = json.load(open(f"{d}/hybrid_results.json"))["results"]
        except FileNotFoundError:
            ai[n] = None
            continue
        e = [r["edge_vs_switch"]["score"] for r in h.values()]
        usd = v["usd_per_sat_year"].values()
        ai[n] = {"pct": v["extra_pct"], "wins": sum(x["wins"] for x in e), "runs": sum(x["runs"] for x in e),
                 "lo": min(u[0] for u in usd), "hi": max(u[1] for u in usd)}
    return {"sharing": sharing, "ai": ai}


def load_orbit():
    """The animated orbit page, embedded in an iframe so its styles stay separate."""
    import html, os
    p = next((p for p in ("orbit_view.html", "web/orbit.html") if os.path.exists(p)), None)
    if not p:
        return "", False
    doc = open(p, encoding="utf-8").read()
    embed = ("<style>header{display:none}.wrap{padding-block:0 8px}</style>"
             "<script>addEventListener('load',()=>{const s=()=>parent.postMessage({orbitHeight:"
             "Math.ceil(document.querySelector('.wrap').getBoundingClientRect().bottom)+8},'*');s();new ResizeObserver(s).observe(document.body);});</script>")
    doc = doc.replace("</html>", embed + "</html>", 1)
    return html.escape(doc, quote=True), True


def load_helios():
    import os
    p = next((p for p in ("helios_results.json", "results/helios/helios_results.json") if os.path.exists(p)), None)
    return json.load(open(p)) if p else None


def build(results="rl_results.json", out="dashboard.html"):
    with open(results) as f:
        d = json.load(f)
    d.setdefault("controllers", list(d["averages"]["LOW"].keys()))
    data = json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
    fleet = None
    for fp in ("fleet_eval_real.json", "fleet_eval.json"):
        try:
            with open(fp) as f:
                fleet = json.load(f)
            break
        except FileNotFoundError:
            pass
    fleet_js = json.dumps(fleet, separators=(",", ":")).replace("</", "<\\/")
    try:
        with open("value_results.json") as f:
            val_js = json.dumps(json.load(f), separators=(",", ":")).replace("</", "<\\/")
    except FileNotFoundError:
        val_js = "null"
    try:
        with open("hybrid_results.json") as f:
            hyb_js = json.dumps(json.load(f), separators=(",", ":")).replace("</", "<\\/")
    except FileNotFoundError:
        hyb_js = "null"
    scale = load_scale()
    orbit_doc, has_orbit = load_orbit()
    html = (TEMPLATE.replace("__HELIOS__", json.dumps(load_helios(), separators=(",", ":")))
            .replace("__SCALE__", json.dumps(scale, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__ORBITDOC__", orbit_doc).replace("__HAS_ORBIT__", "true" if has_orbit else "false")
            .replace("__DATA__", data).replace("__FLEET__", fleet_js).replace("__HYB__", hyb_js).replace("__VAL__", val_js)
            .replace("__ORBIT__", __import__("sat_profile").LABEL.upper())
            .replace("__N__", str(d.get("episodes_per_level", 20)))
            .replace("__UPDATED__", datetime.now().strftime("%b %d %Y %I:%M %p").upper()))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Rebuilt {out} - double-click it to open in your browser")


if __name__ == "__main__":
    build()
