"use strict";

const CLASS_COLOR = {
  transit:  "#4da3ff",
  eclipse:  "#ff6b4a",
  blend:    "#ffb03a",
  variable: "#b07cff",
  noise:    "#7d8b9c",
};

const CLASS_BLURB = {
  transit:  "Planet crossing its host star: shallow, U-shaped, no secondary eclipse.",
  eclipse:  "Eclipsing binary: deep, V-shaped, with a secondary eclipse and alternating depths.",
  blend:    "Deep eclipsing binary diluted by a neighbour in the aperture — planet depth, binary shape.",
  variable: "Starspot rotation or pulsation: smooth and sinusoidal, no sharp ingress.",
  noise:    "No coherent periodic signal above the detection threshold.",
};

// shared Plotly styling so every chart reads as one system
const LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#8b9aad", size: 11,
          family: "ui-monospace, Menlo, monospace" },
  margin: { l: 58, r: 16, t: 10, b: 40 },
  xaxis: { gridcolor: "#1b2531", zerolinecolor: "#243040", linecolor: "#243040" },
  yaxis: { gridcolor: "#1b2531", zerolinecolor: "#243040", linecolor: "#243040" },
  showlegend: false,
  hovermode: "closest",
};
const CONFIG = { displayModeBar: false, responsive: true };

const layout = (over = {}) => ({
  ...structuredClone(LAYOUT),
  ...over,
  xaxis: { ...LAYOUT.xaxis, ...(over.xaxis || {}) },
  yaxis: { ...LAYOUT.yaxis, ...(over.yaxis || {}) },
});

let selection = null;   // {kind:'cached'|'simulate', value:string}
let busy = false;

const $ = (id) => document.getElementById(id);
const fmt = (v, n = 3) =>
  (v === null || v === undefined || !isFinite(v)) ? "—" : Number(v).toFixed(n);

// ---------------------------------------------------------------- bootstrap
async function boot() {
  try {
    const t = await (await fetch("/api/targets")).json();

    $("realList").innerHTML = t.real.map((r) => `
      <button class="target" data-kind="cached" data-value="${r.id}">
        <div class="nm">${r.name}</div>
        <div class="meta">${r.tic} · ${r.n_points.toLocaleString()} pts${
          r.published_period ? ` · P=${r.published_period}d` : ""}</div>
      </button>`).join("") || `<div class="meta" style="color:var(--muted)">
        none cached — run scripts/fetch_real.py</div>`;

    $("simList").innerHTML = t.simulated.map((c) => `
      <button class="target" data-kind="simulate" data-value="${c}">
        <div class="nm" style="color:${CLASS_COLOR[c]}">${c}</div>
        <div class="meta">injected ground truth</div>
      </button>`).join("");

    document.querySelectorAll(".target").forEach((b) =>
      b.addEventListener("click", () => {
        document.querySelectorAll(".target").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        selection = { kind: b.dataset.kind, value: b.dataset.value };
        run();
      }));

    const pill = $("clsPill");
    pill.textContent = t.classifier_loaded ? "classifier loaded" : "no classifier";
    pill.className = "pill " + (t.classifier_loaded ? "on" : "off");
    setFallbackBanner(t.classifier_loaded);
  } catch (e) {
    $("main").innerHTML = `<div class="banner bad">Cannot reach the API: ${e}</div>`;
  }
  pollHealth();
}


// A missing classifier degrades the pipeline to detection + fitting only, and
// the app keeps working -- which is exactly the danger. On unfamiliar hardware
// (a fresh clone on presentation morning, LightGBM failing to find OpenMP) the
// demo would run and quietly answer none of the classification questions a
// judge is there to ask. A status chip is too easy to miss, so this states it
// across the top of the page and says what to do about it.
function setFallbackBanner(loaded) {
  const existing = document.getElementById("fallbackBanner");
  if (loaded) { if (existing) existing.remove(); return; }
  if (existing) return;
  const el = document.createElement("div");
  el.id = "fallbackBanner";
  el.className = "banner bad";
  el.style.margin = "0 0 16px";
  el.innerHTML =
    "<b>Degraded mode &mdash; no classifier loaded.</b> " +
    "Detection, vetting features and transit fitting all still work, but " +
    "nothing on this page is classified: no transit / eclipse / blend call, " +
    "no confidence, no feature drivers. " +
    "Train one with <code>python scripts/train.py</code>, or on Linux check " +
    "that LightGBM found OpenMP (<code>apt-get install libgomp1</code>).";
  const main = document.getElementById("main");
  main.parentNode.insertBefore(el, main);
}

async function pollHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("warmPill").textContent = `cache ${h.warm_entries}`;
    setFallbackBanner(h.classifier_loaded);
    if (h.classifier_loaded) {
      $("clsPill").textContent = "classifier loaded";
      $("clsPill").className = "pill on";
    }
  } catch { /* transient */ }
  setTimeout(pollHealth, 4000);
}

// ------------------------------------------------------------------ analyse
async function run(override) {
  if (busy) return;
  const body = override || (() => {
    if (!selection) return null;
    return selection.kind === "cached"
      ? { cached: selection.value }
      : { simulate: selection.value, seed: parseInt($("seed").value || "42", 10) };
  })();
  if (!body) return;

  body.run_mcmc = $("mcmc").checked;
  body.multi_detrend = $("multi").checked;

  busy = true;
  $("runBtn").disabled = true;
  $("main").innerHTML =
    `<div class="empty"><span class="spinner"></span>Running pipeline&hellip;
     <div style="font-size:12px;margin-top:8px">
       detrend &rarr; BLS &rarr; 22 vetting features${
         body.run_mcmc ? " &rarr; MCMC posterior" : ""}</div></div>`;

  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `HTTP ${r.status}`);
    }
    render(await r.json());
  } catch (e) {
    $("main").innerHTML = `<div class="banner bad">${e.message}</div>`;
  } finally {
    busy = false;
    $("runBtn").disabled = false;
  }
}

// ------------------------------------------------------------------- render
function render(d) {
  const t = d.target;
  const head = `
    <div style="margin-bottom:14px">
      <div style="font-size:17px;font-weight:650">${t.name}</div>
      <div style="color:var(--muted);font-size:12px;margin-top:2px">
        ${t.source} · ${t.n_points.toLocaleString()} points ·
        ${fmt(t.baseline_days, 1)} d baseline · detrender: ${d.detrend_method} ·
        ${d.cached ? "cached" : fmt(d.elapsed_seconds, 1) + " s"}
        ${t.note ? " · " + t.note : ""}
      </div>
    </div>`;

  if (!d.detected) {
    $("main").innerHTML = head +
      `<div class="banner warn">${d.message}</div>` +
      (d.series ? `<div class="panel"><h3>Light curve</h3>
         <div class="chart" id="cRaw" style="height:300px"></div></div>` : "");
    if (d.series) drawRaw(d);
    return;
  }

  const det = d.detection, cls = d.classification, fit = d.fit;
  const col = cls ? (CLASS_COLOR[cls.label] || "#4da3ff") : "#4da3ff";

  // truth / published comparison banner
  let banner = "";
  if (t.truth_label && cls) {
    const ok = t.truth_label === cls.label;
    banner = `<div class="banner ${ok ? "ok" : "bad"}">
      Injected truth: <b>${t.truth_label}</b> — prediction
      ${ok ? "matches" : `differs (<b>${cls.label}</b>)`}.</div>`;
  } else if (t.published_period) {
    const ratio = det.period_days / t.published_period;
    const harm = [1, 2, 0.5, 3, 1 / 3].find((m) => Math.abs(ratio / m - 1) < 0.02);
    const tag = harm === 1 ? "matches exactly"
              : harm ? `is the ${harm === 0.5 ? "½" : harm}× harmonic of`
              : "does NOT match";
    banner = `<div class="banner ${harm === 1 ? "ok" : harm ? "warn" : "bad"}">
      Recovered period <b>${fmt(det.period_days, 5)} d</b> ${tag}
      the published <b>${t.published_period} d</b>.</div>`;
  }

  // The classifier and the fit disagree. This sits above the classification
  // card deliberately: the whole failure mode it guards against is a reader
  // seeing a confident label and stopping there.
  let caution = "";
  if (d.caution_flag) {
    caution = `<div class="banner bad">
      <b>Caution — treat this classification with suspicion.</b><br>
      ${d.caution_reason || "the fit does not describe the data"}</div>`;
  }

  // The retained detection came from the fallback detrender, so its SDE is the
  // better of two trials rather than a single measurement.
  let detrendNote = "";
  if (d.detrend_is_fallback && d.sde_corrected != null) {
    detrendNote = `<div class="banner warn">
      Detected by the fallback <b>${d.detrend_method}</b> detrender, not the
      primary one. Across ${d.n_detrend_trials} trials the SDE of
      <b>${fmt(det.sde, 2)}</b> is worth <b>${fmt(d.sde_corrected, 2)}</b> after
      a look-elsewhere discount.</div>`;
  }

  $("main").innerHTML = head + caution + banner + detrendNote + `
    <div class="metrics">
      <div class="card"><div class="k">Classification</div>
        <div class="v" style="color:${col}">${cls ? cls.label : "—"}</div>
        <div class="s">${cls ? (cls.confidence * 100).toFixed(1) + "% confidence" : "no classifier"}</div></div>
      <div class="card"><div class="k">Orbital period</div>
        <div class="v">${fmt(det.period_days, 4)}<span style="font-size:13px;color:var(--muted)"> d</span></div>
        <div class="s">${det.n_transits} transits observed</div></div>
      <div class="card"><div class="k">Transit depth</div>
        <div class="v">${fmt(det.depth_ppm, 0)}<span style="font-size:13px;color:var(--muted)"> ppm</span></div>
        <div class="s">duration ${fmt(det.duration_hours, 2)} h</div></div>
      <div class="card"><div class="k">Significance</div>
        <div class="v">${fmt(det.sde, 1)}<span style="font-size:13px;color:var(--muted)"> SDE</span></div>
        <div class="s">S/N ${fmt(det.snr, 0)}</div></div>
    </div>

    ${cls ? `<div class="panel"><h3>Classification</h3>
      <div class="chart" id="cProb" style="height:190px"></div>
      <div style="color:var(--muted);font-size:12px;margin-top:6px">${CLASS_BLURB[cls.label] || ""}</div>
      ${cls.drivers && cls.drivers.length ? `<div class="drivers">
        <span style="color:var(--muted);font-size:11px;align-self:center">main evidence:</span>
        ${cls.drivers.map(([k, v]) =>
          `<span class="chip"><b>${k}</b> ${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`).join("")}
      </div>` : ""}</div>` : ""}

    <div class="panel"><h3>Light curve and detrending</h3>
      <div class="chart" id="cRaw" style="height:250px"></div>
      <div class="chart" id="cDet" style="height:220px"></div></div>

    <div class="row2">
      <div class="panel"><h3>BLS periodogram</h3>
        <div class="chart" id="cPg" style="height:250px"></div></div>
      <div class="panel"><h3>Phase-folded</h3>
        <div class="chart" id="cFold" style="height:250px"></div></div>
    </div>

    <div class="panel"><h3>Vetting diagnostics</h3>
      <div class="row2">
        <div><div class="chart" id="cOE" style="height:230px"></div>
          <div style="color:var(--muted);font-size:11px;text-align:center">
            Odd vs even transits — they separate for an eclipsing binary</div></div>
        <div><div class="chart" id="c2P" style="height:230px"></div>
          <div style="color:var(--muted);font-size:11px;text-align:center">
            Folded at 2×P — a planet shows two equal events, an EB a shallower secondary</div></div>
      </div></div>

    ${fit ? fitPanel(fit, t) : ""}

    <div class="panel"><details><summary>Vetting feature vector (${
      d.features ? Object.keys(d.features).length : 0})</summary>
      <table><thead><tr><th>feature</th><th>value</th></tr></thead><tbody>
      ${Object.entries(d.features || {}).map(([k, v]) =>
        `<tr><td class="mono">${k}</td><td class="num">${fmt(v, 4)}</td></tr>`).join("")}
      </tbody></table></details></div>`;

  drawRaw(d); drawDet(d); drawPg(d); drawFold(d); drawOE(d); draw2P(d);
  if (cls) drawProb(cls);
}

function fitPanel(f, t) {
  const pm = (v, e, n) => e === null || e === undefined
    ? fmt(v, n) : `${fmt(v, n)} <span style="color:var(--muted)">± ${fmt(e, n)}</span>`;
  const cmp = (v, p, n) => p ? `${fmt(p, n)} <span style="color:var(--muted)">(${
    ((v / p - 1) * 100 >= 0 ? "+" : "")}${((v / p - 1) * 100).toFixed(1)}%)</span>` : "—";
  return `<div class="panel"><h3>Fitted transit model</h3>
    ${f.warning ? `<div class="banner warn">${f.warning}</div>` : ""}
    <table><thead><tr><th>parameter</th><th>value ± 1σ</th><th>unit</th>
      <th>published</th></tr></thead><tbody>
      <tr><td>Orbital period</td><td class="num">${pm(f.period_days, f.period_err, 5)}</td>
        <td>days</td><td class="num">${t.published_period ? fmt(t.published_period, 5) : "—"}</td></tr>
      <tr><td>Transit depth (observed)</td>
        <td class="num">${pm(f.depth_observed_ppm, f.depth_observed_err_ppm, 0)}</td>
        <td>ppm</td><td>—</td></tr>
      <tr><td>Transit depth (geometric, Rp²/R*²)</td>
        <td class="num">${pm(f.depth_geometric_ppm, f.depth_geometric_err_ppm, 0)}</td>
        <td>ppm</td><td>—</td></tr>
      <tr><td>Transit duration</td>
        <td class="num">${pm(f.duration_hours, f.duration_err_hours, 3)}</td>
        <td>hours</td><td>—</td></tr>
      <tr><td>Rp / R*</td><td class="num">${pm(f.rp_over_rs, f.rp_over_rs_err, 5)}</td>
        <td>—</td><td class="num">${cmp(f.rp_over_rs, t.published_rp_rs, 5)}</td></tr>
      <tr><td>Impact parameter</td><td class="num">${fmt(f.impact_param, 3)}</td><td>—</td><td>—</td></tr>
      <tr><td>Reduced χ²</td><td class="num" style="color:${
        f.reliable ? "var(--good)" : "var(--bad)"}">${fmt(f.chi2_reduced, 2)}</td>
        <td>—</td><td>—</td></tr>
    </tbody></table>
    <div style="color:var(--muted);font-size:11px;margin-top:9px">
      Uncertainties are the standard deviation of the MCMC posterior (32 walkers).
      Observed depth exceeds the geometric depth because limb darkening makes the
      stellar disc brighter at its centre.</div></div>`;
}

// -------------------------------------------------------------------- charts
const sc = (s, o = {}) => ({
  x: s.x, y: s.y, type: "scattergl", mode: "markers",
  marker: { size: 2, color: "#5a6b80" }, hoverinfo: "skip", ...o,
});

function drawRaw(d) {
  const s = d.series, tr = [sc(s.raw, { name: "raw" })];
  if (s.trend.x.length)
    tr.push({ x: s.trend.x, y: s.trend.y, type: "scattergl", mode: "lines",
              line: { color: "#ff6b4a", width: 1.4 }, hoverinfo: "skip" });
  Plotly.newPlot("cRaw", tr,
    layout({ yaxis: { ...LAYOUT.yaxis, title: "raw flux" },
             xaxis: { ...LAYOUT.xaxis, title: "" }, margin: { l: 58, r: 16, t: 6, b: 22 } }),
    CONFIG);
}

function drawDet(d) {
  const s = d.series, det = d.detection;
  const shapes = [];
  // Mark each predicted transit. Capped: a pathologically short period would
  // otherwise push thousands of shapes and lock up the browser mid-demo.
  if (det && det.period_days > 0) {
    const MAX_MARKS = 200;
    const x0 = s.detrended.x[0], x1 = s.detrended.x[s.detrended.x.length - 1];
    const k0 = Math.floor((x0 - det.t0) / det.period_days);
    const kN = Math.ceil((x1 - det.t0) / det.period_days);
    if (kN - k0 <= MAX_MARKS) {
      for (let k = k0; k <= kN; k++) {
        const tc = det.t0 + k * det.period_days;
        if (tc >= x0 && tc <= x1) shapes.push({
          type: "line", x0: tc, x1: tc, y0: 0, y1: 1, yref: "paper",
          line: { color: "#4da3ff", width: 1, dash: "dot" } });
      }
    }
  }
  Plotly.newPlot("cDet", [sc(s.detrended, { marker: { size: 2, color: "#8fa3ba" } })],
    layout({ shapes, yaxis: { ...LAYOUT.yaxis, title: "detrended" },
             xaxis: { ...LAYOUT.xaxis, title: "time (days)" },
             margin: { l: 58, r: 16, t: 6, b: 38 } }), CONFIG);
}

function drawPg(d) {
  const s = d.series, P = d.detection.period_days;
  const shapes = [{ type: "line", x0: Math.log10(P), x1: Math.log10(P), y0: 0, y1: 1,
                    yref: "paper", line: { color: "#ff6b4a", width: 1.4, dash: "dash" } }];
  [[2, "2P"], [0.5, "P/2"]].forEach(([m]) => {
    const p = P * m;
    if (p >= Math.min(...s.periodogram.x) && p <= Math.max(...s.periodogram.x))
      shapes.push({ type: "line", x0: Math.log10(p), x1: Math.log10(p), y0: 0, y1: 1,
                    yref: "paper", line: { color: "#ffb03a", width: 1, dash: "dot" } });
  });
  Plotly.newPlot("cPg",
    [{ x: s.periodogram.x, y: s.periodogram.y, type: "scattergl", mode: "lines",
       line: { color: "#4da3ff", width: 1 }, hovertemplate: "P=%{x:.4f} d<extra></extra>" }],
    layout({ shapes, xaxis: { ...LAYOUT.xaxis, title: "period (days)", type: "log" },
             yaxis: { ...LAYOUT.yaxis, title: "BLS power" } }), CONFIG);
}

function drawFold(d) {
  const s = d.series, det = d.detection;
  const half = Math.min(0.5, 3 * (det.duration_hours / 24) / det.period_days);
  const tr = [
    sc(s.fold, { marker: { size: 2, color: "#3d4a5c" } }),
    { x: s.fold_binned.x, y: s.fold_binned.y, type: "scatter", mode: "markers",
      marker: { size: 5, color: "#e6edf3" },
      error_y: { type: "data", array: s.fold_binned.e, width: 0, color: "#5a6b80" },
      hovertemplate: "phase %{x:.4f}<br>%{y:.6f}<extra></extra>" },
  ];
  if (s.model)
    tr.push({ x: s.model.x, y: s.model.y, type: "scatter", mode: "lines",
              line: { color: "#ff6b4a", width: 2 }, hoverinfo: "skip" });
  Plotly.newPlot("cFold", tr,
    layout({ xaxis: { ...LAYOUT.xaxis, title: "phase", range: [-half, half] },
             yaxis: { ...LAYOUT.yaxis, title: "normalised flux" } }), CONFIG);
}

function drawOE(d) {
  const s = d.series, tr = [];
  if (s.even) tr.push({ x: s.even.x, y: s.even.y, type: "scatter", mode: "lines+markers",
                        line: { color: "#4da3ff", width: 1.6 }, marker: { size: 3 }, name: "even" });
  if (s.odd) tr.push({ x: s.odd.x, y: s.odd.y, type: "scatter", mode: "lines+markers",
                       line: { color: "#ff6b4a", width: 1.6 }, marker: { size: 3 }, name: "odd" });
  Plotly.newPlot("cOE", tr,
    layout({ showlegend: true,
             legend: { orientation: "h", y: 1.16, x: 0, font: { size: 10 } },
             xaxis: { ...LAYOUT.xaxis, title: "hours from mid-transit" },
             yaxis: { ...LAYOUT.yaxis, title: "flux" } }), CONFIG);
}

function draw2P(d) {
  const s = d.series;
  const shapes = [0, 0.5, -0.5].map((x) => ({
    type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper",
    line: { color: x === 0 ? "#4da3ff" : "#ff6b4a", width: 1, dash: "dot" } }));
  Plotly.newPlot("c2P", [sc(s.fold_2p, { marker: { size: 3, color: "#8fa3ba" } })],
    layout({ shapes, xaxis: { ...LAYOUT.xaxis, title: "phase (period = 2P)" },
             yaxis: { ...LAYOUT.yaxis, title: "flux" } }), CONFIG);
}

function drawProb(cls) {
  const labels = Object.keys(cls.probabilities);
  const vals = labels.map((k) => cls.probabilities[k]);
  Plotly.newPlot("cProb",
    [{ x: vals, y: labels, type: "bar", orientation: "h",
       marker: { color: labels.map((k) => CLASS_COLOR[k]) },
       text: vals.map((v) => (v * 100).toFixed(1) + "%"),
       textposition: "outside", textfont: { color: "#8b9aad", size: 10 },
       hoverinfo: "skip" }],
    layout({ xaxis: { ...LAYOUT.xaxis, range: [0, 1.18], title: "calibrated probability" },
             yaxis: { ...LAYOUT.yaxis, automargin: true },
             margin: { l: 70, r: 16, t: 6, b: 34 } }), CONFIG);
}

// -------------------------------------------------------------------- events
$("runBtn").addEventListener("click", () => run());
$("ticBtn").addEventListener("click", () => {
  const v = $("ticInput").value.trim();
  if (v) {
    document.querySelectorAll(".target").forEach((x) => x.classList.remove("active"));
    selection = null;
    run({ tic: v });
  }
});
$("ticInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("ticBtn").click(); });

boot();
