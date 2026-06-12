"use strict";
const $ = s => document.querySelector(s);
let DATA = null, PHASE = "before", AVAIL = {};

// display names: FORGE is the umbrella; each experiment has a model + task label
const DISPLAY = {
  genre:  "BEARDOWN · genre recognition",
  phonon: "phonon · DOS reproduction",
  atlas:  "atlas · Run 3",
};

// cache-bust on the EDA generation timestamp so regenerated figures never serve stale
const figURL = file =>
  `/figures/${PHASE}/${file}?v=${encodeURIComponent((DATA && DATA.generated) || "")}`;

async function init(){
  try{
    const c = await (await fetch("/api/config")).json();
    PHASE = c.default_phase; AVAIL = c.available || {};
    $("#expName").textContent = DISPLAY[c.experiment] || (c.experiment || "experiment").toUpperCase();
    if (c.logo){ const el = $("#expLogo"); el.src = c.logo; el.alt = c.experiment; el.hidden = false; }
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
  ["#hero","#integrityPanel","#featStats","#varDetail","#featTable"].forEach(s=>$(s).innerHTML="");
  $("#typeTable").innerHTML="";
}

function render(){
  $("#phaseBadge").textContent = DATA.phase || PHASE;
  $("#gen").textContent = DATA.generated ? "generated " + DATA.generated : "";
  renderCards(); renderHero(); renderIntegrity(); renderTypes(); renderFeatures();
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

// ---- numeric feature statistics: cards + variable toggle + sortable table ---
const COLS = [
  {k:"feature", label:"feature"}, {k:"mean", label:"mean"}, {k:"median", label:"median"},
  {k:"std", label:"std"}, {k:"min", label:"min"}, {k:"max", label:"max"},
  {k:"iqr", label:"IQR"}, {k:"n_outliers_iqr", label:"outliers"}, {k:"skew", label:"skew"},
];
let SORT = {k:"feature", dir:1};
let SEL = null;

const fnum = v => {
  if (v == null || isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-2)) return v.toExponential(2);
  return (Math.round(v * 1000) / 1000).toString();
};
const figForFeature = f => (DATA.figures||[]).find(g=>g.kind==="feature_dist" && g.feature===f);

function renderFeatures(){
  const per = DATA.nerd_stats?.per_feature || {};
  const names = Object.keys(per);
  if (!names.length){ ["#featStats","#varDetail","#featTable"].forEach(s=>$(s).innerHTML=""); return; }

  // dashboard statistics across all numeric features
  let totOut=0, skewName=names[0], iqrName=names[0], flagged=0;
  for (const n of names){
    const p = per[n];
    totOut += p.n_outliers_iqr || 0;
    if (Math.abs(p.skew) > Math.abs(per[skewName].skew)) skewName = n;
    if ((p.iqr||0) > (per[iqrName].iqr||0)) iqrName = n;
    if ((p.outlier_pct||0) > 5) flagged++;
  }
  $("#featStats").innerHTML = [
    ["numeric features", names.length], ["total outliers", totOut],
    [">5% outliers", flagged], ["most skewed", skewName], ["widest IQR", iqrName],
  ].map(([l,n])=>`<div class="card"><div class="n" style="font-size:18px">${n}</div><div class="l">${l}</div></div>`).join("");

  $("#varSelect").innerHTML = names.map(n=>`<option value="${n}">${n}</option>`).join("");
  $("#varSelect").onchange = e => selectVar(e.target.value);
  $("#search").oninput = () => drawTable();

  drawTable();
  selectVar(SEL && per[SEL] ? SEL : names[0]);
}

function drawTable(){
  const per = DATA.nerd_stats?.per_feature || {};
  const q = ($("#search").value || "").toLowerCase();
  let rows = Object.entries(per).map(([feature, p]) => ({feature, ...p}))
                   .filter(r => !q || r.feature.toLowerCase().includes(q));
  rows.sort((a,b)=>{
    const x=a[SORT.k], y=b[SORT.k];
    const c = (SORT.k==="feature") ? String(x).localeCompare(String(y)) : (x-y);
    return c * SORT.dir;
  });
  $("#figCount").textContent = rows.length + " / " + Object.keys(per).length;
  const head = COLS.map(c=>{
    const arrow = SORT.k===c.k ? (SORT.dir>0?" ▲":" ▼") : "";
    return `<th data-k="${c.k}" class="sortable">${c.label}${arrow}</th>`;
  }).join("");
  const body = rows.map(r=>{
    const cls = r.feature===SEL ? ' class="sel"' : "";
    const tds = COLS.map(c => c.k==="feature" ? `<td>${r.feature}</td>`
      : c.k==="n_outliers_iqr" ? `<td>${r.n_outliers_iqr} <span class="dim">(${r.outlier_pct}%)</span></td>`
      : `<td>${fnum(r[c.k])}</td>`).join("");
    return `<tr data-f="${r.feature}"${cls}>${tds}</tr>`;
  }).join("");
  $("#featTable").innerHTML = `<tr>${head}</tr>${body}`;
}

function selectVar(name){
  const per = DATA.nerd_stats?.per_feature || {};
  const p = per[name]; if(!p) return;
  SEL = name;
  if ($("#varSelect").value !== name) $("#varSelect").value = name;
  const fig = figForFeature(name);
  const stats = [
    ["count", p.count], ["mean", fnum(p.mean)], ["median", fnum(p.median)],
    ["mode≈", fnum(p.mode_round3)], ["std", fnum(p.std)], ["min", fnum(p.min)],
    ["max", fnum(p.max)], ["Q1", fnum(p.q1)], ["Q3", fnum(p.q3)], ["IQR", fnum(p.iqr)],
    ["outliers", `${p.n_outliers_iqr} (${p.outlier_pct}%)`], ["skew", fnum(p.skew)],
  ];
  $("#varDetail").innerHTML = `
    <div class="var-fig">${fig
      ? `<img src="${figURL(fig.file)}" alt="${name}" data-file="${fig.file}">`
      : `<div class="dim">no figure</div>`}</div>
    <div class="stat-grid">${stats.map(([k,v])=>
      `<div class="stat"><span class="sk">${k}</span><span class="sv">${v}</span></div>`).join("")}</div>`;
  document.querySelectorAll("#featTable tr[data-f]").forEach(tr=>
    tr.classList.toggle("sel", tr.dataset.f===name));
}

// table: sort on header click, select variable on row click
$("#featTable").addEventListener("click", e=>{
  const th = e.target.closest("th.sortable");
  if (th){ const k=th.dataset.k; SORT = {k, dir: SORT.k===k ? -SORT.dir : 1}; drawTable(); return; }
  const tr = e.target.closest("tr[data-f]");
  if (tr){ selectVar(tr.dataset.f); $("#varDetail").scrollIntoView({behavior:"smooth", block:"nearest"}); }
});

// enlarge the selected variable's figure
$("#varDetail").addEventListener("click", e=>{
  const img = e.target.closest("img"); if(!img) return;
  $("#modalImg").src = img.src; $("#modalCap").textContent = SEL || "";
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
