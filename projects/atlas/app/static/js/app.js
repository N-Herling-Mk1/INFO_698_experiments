"use strict";
const $ = s => document.querySelector(s);
let DATA = null, PHASE = "before", AVAIL = {};

const figURL = file => `/figures/${PHASE}/${file}`;

async function init(){
  try{
    const c = await (await fetch("/api/config")).json();
    PHASE = c.default_phase; AVAIL = c.available || {};
    $("#expName").textContent = (c.experiment || "experiment").toUpperCase();
    buildPhaseToggle(c.phases || ["before","after"]);
  }catch(e){ /* config optional; fall back to before */ }
  await load();
}

function buildPhaseToggle(phases){
  $("#phaseToggle").innerHTML = phases.map(p=>{
    const dis = AVAIL[p] ? "" : "disabled";
    const act = p===PHASE ? "active" : "";
    return `<button class="${act}" data-phase="${p}" ${dis}>${p}</button>`;
  }).join("");
  $("#phaseToggle").querySelectorAll("button").forEach(b=>{
    b.onclick = () => { if(b.disabled) return; PHASE=b.dataset.phase; syncToggle(); load(); };
  });
}
function syncToggle(){
  $("#phaseToggle").querySelectorAll("button").forEach(b=>
    b.classList.toggle("active", b.dataset.phase===PHASE));
}

async function load(){
  let r;
  try{ r = await fetch(`/api/eda?phase=${PHASE}`); }
  catch(e){ return fail("backend unreachable"); }
  if(!r.ok){ const j = await r.json().catch(()=>({})); return fail(j.error || ("HTTP "+r.status)); }
  DATA = await r.json();
  render();
}
function fail(msg){
  $("#cards").innerHTML = `<div class="err">⚠ ${msg}<br><span class="dim">reload after generating this snapshot</span></div>`;
  ["#hero","#integrityPanel","#gallery"].forEach(s=>$(s).innerHTML="");
  $("#typeTable").innerHTML="";
}

function render(){
  $("#phaseBadge").textContent = DATA.phase || PHASE;
  $("#gen").textContent = DATA.generated ? "generated " + DATA.generated : "";
  renderCards(); renderHero(); renderIntegrity(); renderTypes(); renderFigures();
}

function renderCards(){
  const mc = DATA.missing_corrupt || {}, c = mc.counts || {};
  const cards = [
    ["wav files", c.wav_total ?? "—"],
    ["spectrograms", c.grey_total ?? "—"],
    ["features", DATA.nerd_stats?.n_features ?? "—"],
    ["known issues", (mc.known_issues||[]).length],
    ["corrupt", (mc.corrupt_audio||[]).length],
    ["figures", (DATA.figures||[]).length],
  ];
  $("#cards").innerHTML = cards.map(([l,n])=>
    `<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
}

function renderHero(){
  const figs = DATA.figures || [];
  const pick = k => figs.find(f=>f.kind===k);
  const heroes = [pick("class_balance"), pick("exemplars")].filter(Boolean);
  $("#hero").innerHTML = heroes.map(f=>
    `<div class="imgwrap"><img src="${figURL(f.file)}" alt="${f.kind}"></div>`).join("")
    || `<div class="dim">no hero figures emitted</div>`;
}

function renderIntegrity(){
  const mc = DATA.missing_corrupt || {};
  const sec = (title, rows, cols) => {
    if(!rows || !rows.length) return `<h2>${title}</h2><div class="dim">none</div>`;
    const head = cols.map(c=>`<th>${c}</th>`).join("");
    const body = rows.map(r=>`<tr>${cols.map(c=>`<td>${r[c]??""}</td>`).join("")}</tr>`).join("");
    return `<h2>${title} <span class="dim">(${rows.length})</span></h2><table><tr>${head}</tr>${body}</table>`;
  };
  const miss = Object.entries(mc.missing_per_representation||{})
    .flatMap(([rep,ids]) => ids.map(id => ({id, representation:rep})));
  let html = "";
  html += sec("Corrupt audio", mc.corrupt_audio, ["id","genre","error","bytes"]);
  html += sec("Off-duration", mc.off_duration, ["id","genre","duration_s","sample_rate"]);
  html += sec("Segment anomalies (3s)", mc.segment_anomalies, ["track","segments"]);
  html += sec("Missing across representations", miss, ["id","representation"]);
  html += `<h2>Known issues <span class="dim">(${(mc.known_issues||[]).length})</span></h2>`;
  html += (mc.known_issues||[]).map(s=>`<div class="issue">${s}</div>`).join("");
  $("#integrityPanel").innerHTML = html;
}

function renderTypes(){
  const cols = DATA.type_audit?.columns || [];
  const rows = cols.map(a=>`<tr>
      <td>${a.column}</td><td class="dim">${a.expected}</td><td>${a.actual_dtype}</td>
      <td>${a.match ? '<span class="pill ok">match</span>' : '<span class="pill no">check</span>'}</td>
      <td class="dim">${a.note||""}</td></tr>`).join("");
  let html = `<tr><th>column</th><th>expected</th><th>actual</th><th>type</th><th>note</th></tr>${rows}`;
  const sp = DATA.type_audit?.spectrograms;
  if(sp){
    html += `<tr><td colspan="5" class="dim">spectrograms · modes ${JSON.stringify(sp.observed_modes)} · sizes ${
      JSON.stringify(sp.observed_sizes)} · unreadable ${(sp.unreadable||[]).length}</td></tr>`;
  }
  $("#typeTable").innerHTML = html;
}

function renderFigures(){
  const feats = (DATA.figures||[]).filter(f=>f.kind==="feature_dist");
  const draw = q => {
    const list = feats.filter(f=>!q || f.feature.toLowerCase().includes(q.toLowerCase()));
    $("#figCount").textContent = list.length + " / " + feats.length;
    $("#gallery").innerHTML = list.map(f=>`
      <div class="tile" data-file="${f.file}" data-cap="${(f.caption||f.feature).replace(/"/g,'&quot;')}">
        <img loading="lazy" src="${figURL(f.file)}" alt="${f.feature}">
        <div class="cap"><b>${f.feature}</b><br>${f.caption||""}</div></div>`).join("");
  };
  draw($("#search").value || "");
  $("#search").oninput = e => draw(e.target.value);
}

// figure modal
$("#gallery").addEventListener("click", e=>{
  const t = e.target.closest(".tile"); if(!t) return;
  $("#modalImg").src = figURL(t.dataset.file);
  $("#modalCap").textContent = t.dataset.cap;
  $("#modal").classList.add("open");
});
$("#modal").addEventListener("click", ()=>$("#modal").classList.remove("open"));

// tabs
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("nav button").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll("main section").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); $("#"+b.dataset.tab).classList.add("active");
});

init();
