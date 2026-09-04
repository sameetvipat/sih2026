"use strict";

/* ============================================================================
   Transit Console frontend.

   Design tokens live in style.css and nowhere else. Everything here -- including
   the Plotly themes -- reads its colours back out of the stylesheet, so a
   retheme is a single edit to :root rather than a sweep through this file.
   ========================================================================= */

const css = (n) =>
  getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const CLASSES = ["transit", "eclipse", "blend", "variable", "noise"];
const CLASS_COLOR = {};
CLASSES.forEach((c) => (CLASS_COLOR[c] = css("--" + c)));

const T = {
  text:   css("--text"),
  text2:  css("--text-2"),
  muted:  css("--muted"),
  muted2: css("--muted-2"),
  line:   css("--line"),
  grid:   css("--line-soft"),
  brand:  css("--brand"),
  blend:  css("--blend"),
  eclipse:css("--eclipse"),
  good:   css("--good"),
  bad:    css("--bad"),
  mono:   '"Share Tech Mono", ui-monospace, Menlo, monospace',
};

const CLASS_BLURB = {
  transit:  "Planet crossing its host star: shallow, U-shaped, no secondary eclipse.",
  eclipse:  "Eclipsing binary: deep, V-shaped, with a secondary eclipse and alternating depths.",
  blend:    "Deep eclipsing binary diluted by a neighbour in the aperture — planet depth, binary shape.",
  variable: "Starspot rotation or pulsation: smooth and sinusoidal, no sharp ingress.",
  noise:    "No coherent periodic signal above the detection threshold.",
};

/* The shape each class makes in a folded light curve. Showing it next to the
   name teaches the vocabulary the classifier keys on before anything is run. */
const CLASS_GLYPH = {
  // flat, square-shouldered dip -- the planet is small, so ingress is abrupt
  transit:  "M1 3h8c.6 0 .8 8 2 8h4c1.2 0 1.4-8 2-8h9",
  // deep, sharp V straight through the minimum
  eclipse:  "M1 2h8l6 11 6-11h9",
  // V shape at transit depth: the whole point of the class
  blend:    "M1 3h8l6 5.5 6-5.5h9",
  variable: "M1 7c3.5-6 6.5-6 10 0s6.5 6 10 0 4-4 9-1.5",
  noise:    "M1 7l2-2.5 2 4 2-4.5 2 5 2-3.5 2 3 2-4.5 2 4 2-2.5 2 3 2-2.5 2 2",
};
const glyph = (c) => CLASS_GLYPH[c]
  ? `<span class="glyph"><svg width="32" height="14" viewBox="0 0 32 14" fill="none"
       aria-hidden="true"><path d="${CLASS_GLYPH[c]}" stroke="currentColor"
       stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg></span>`
  : `<span class="glyph"></span>`;

const ICON = {
  bad:  `<svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
           <path d="M8 1.6 15 14.4H1L8 1.6Z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>
           <path d="M8 6v3.4M8 11.6v.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  warn: `<svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
           <circle cx="8" cy="8" r="6.6" stroke="currentColor" stroke-width="1.3"/>
           <path d="M8 4.4v4.2M8 11v.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
  ok:   `<svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
           <circle cx="8" cy="8" r="6.6" stroke="currentColor" stroke-width="1.3"/>
           <path d="m5 8.2 2.2 2.2L11 6" stroke="currentColor" stroke-width="1.5"
             stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};

const banner = (kind, html) =>
  `<div class="banner ${kind}"><span class="rail"></span>
     <span class="ico">${ICON[kind] || ICON.warn}</span><span>${html}</span></div>`;

/* ---------------------------------------------------------- Plotly theming */
const LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: T.muted, size: 10, family: T.mono },
  margin: { l: 56, r: 14, t: 8, b: 38 },
  xaxis: { gridcolor: T.grid, zerolinecolor: T.line, linecolor: T.line,
           tickcolor: T.line, ticks: "outside", ticklen: 3 },
  yaxis: { gridcolor: T.grid, zerolinecolor: T.line, linecolor: T.line,
           tickcolor: T.line, ticks: "outside", ticklen: 3 },
  showlegend: false,
  hovermode: "closest",
  hoverlabel: { bgcolor: css("--panel-2"), bordercolor: T.line,
                font: { family: T.mono, size: 11, color: T.text } },
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

/* ============================================================== stepper === */
const STAGES = ["detrend", "bls", "features", "classify", "fit"];

function setStage(name) {
  const i = STAGES.indexOf(name);
  $("stepper").querySelectorAll("b").forEach((b, k) => {
    b.classList.toggle("done", i < 0 ? false : k < i);
    b.classList.toggle("active", k === i);
  });
  document.querySelectorAll(".stage").forEach((el, k) => {
    el.classList.toggle("done", i < 0 ? false : k < i);
    el.classList.toggle("active", k === i);
  });
}
function clearStages() {
  $("stepper").querySelectorAll("b").forEach((b) => b.className = "");
}

/* ============================================================== bootstrap = */
async function boot() {
  try {
    const t = await (await fetch("/api/targets")).json();

    $("realList").innerHTML = t.real.map((r) => `
      <button class="target" data-kind="cached" data-value="${r.id}">
        <span class="rail"></span>
        <span><span class="nm">${r.name}</span>
          <span class="meta">${r.tic} · ${r.n_points.toLocaleString()}&nbsp;pts${
            r.published_period ? ` · P=${r.published_period}&nbsp;d` : ""}</span></span>
        ${glyph()}
      </button>`).join("") ||
      `<div class="meta" style="color:var(--muted);font-family:var(--f-data);font-size:var(--t-11)">
         none cached — run scripts/fetch_real.py</div>`;

    $("simList").innerHTML = t.simulated.map((c) => `
      <button class="target" data-kind="simulate" data-value="${c}"
              data-class="${c}" style="--cls:${CLASS_COLOR[c]}">
        <span class="rail"></span>
        <span><span class="nm">${c}</span>
          <span class="meta">injected ground truth</span></span>
        ${glyph(c)}
      </button>`).join("");

    document.querySelectorAll(".target").forEach((b) =>
      b.addEventListener("click", () => {
        document.querySelectorAll(".target").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        selection = { kind: b.dataset.kind, value: b.dataset.value };
        document.body.classList.remove("nav-open");
        run();
      }));

    const pill = $("clsPill");
    pill.textContent = t.classifier_loaded ? "classifier loaded" : "no classifier";
    pill.className = "pill " + (t.classifier_loaded ? "on" : "off");
    setFallbackBanner(t.classifier_loaded);

    /* Deep link: ?simulate=blend or ?cached=261136679 runs that target on load,
       so a specific result can be bookmarked and reopened during a demo without
       clicking through to it. ?seed= pins the simulated draw. */
    const q = new URLSearchParams(location.search);
    if (q.get("seed")) $("seed").value = q.get("seed");
    const deep = q.get("simulate") ? { kind: "simulate", value: q.get("simulate") }
               : q.get("cached")   ? { kind: "cached",   value: q.get("cached") }
               : null;
    const btn = deep && document.querySelector(
      `.target[data-kind="${deep.kind}"][data-value="${CSS.escape(deep.value)}"]`);
    if (btn) { btn.click(); } else { showEmpty(); }
  } catch (e) {
    $("result").innerHTML = banner("bad", `Cannot reach the API: ${e}`);
  }
  pollHealth();
}

/* The first thing anyone sees. A line of grey text wastes it, so the zero state
   draws the geometry the whole app is about and says what the tool does. */
function showEmpty() {
  $("result").innerHTML = `
    <div class="empty">
      <svg width="320" height="150" viewBox="0 0 320 150" fill="none" aria-hidden="true">
        <circle cx="86" cy="38" r="27" fill="var(--brand)" opacity=".12"/>
        <circle cx="86" cy="38" r="27" fill="none" stroke="var(--brand)" stroke-width="1.4"/>
        <path d="M48 38h76" stroke="var(--line-hi)" stroke-width="1" stroke-dasharray="3 4"/>
        <circle cx="98" cy="38" r="7" fill="var(--void)" stroke="var(--blend)" stroke-width="1.5"/>
        <path d="M98 47v50" stroke="var(--line-hi)" stroke-width="1" stroke-dasharray="3 4"/>
        <path d="M40 106v-8M40 106h250" stroke="var(--line)" stroke-width="1"/>
        <path d="M44 97h40c1.6 0 2 16 5 16h18c3 0 3.4-16 5-16h174"
          stroke="var(--brand)" stroke-width="1.8"
          stroke-linecap="round" stroke-linejoin="round"/>
        <text x="30" y="78" fill="var(--muted-2)" font-family="Share Tech Mono, monospace"
          font-size="9" letter-spacing="1.4" text-anchor="middle"
          transform="rotate(-90 30 78)">FLUX</text>
        <text x="165" y="124" fill="var(--muted-2)" font-family="Share Tech Mono, monospace"
          font-size="9" letter-spacing="1.4" text-anchor="middle">TIME</text>
      </svg>
      <h2>Select a target</h2>
      <p>Pick a cached TESS light curve or a simulated signal with known injected
      truth. The pipeline detrends it, searches for a period with BLS, computes
      the vetting diagnostics, classifies the dip and fits a transit model with
      MCMC uncertainties.</p>
    </div>`;
  clearStages();
}

/* A missing classifier degrades the pipeline to detection + fitting only, and
   the app keeps working -- which is exactly the danger. On unfamiliar hardware
   (a fresh clone on presentation morning, LightGBM failing to find OpenMP) the
   demo would run and quietly answer none of the classification questions a
   judge is there to ask. A status chip is too easy to miss, so this states it
   across the top of the page and says what to do about it. */
function setFallbackBanner(loaded) {
  const host = $("globalAlerts");
  const existing = document.getElementById("fallbackBanner");
  if (loaded) { if (existing) existing.remove(); return; }
  if (existing) return;
  const el = document.createElement("div");
  el.id = "fallbackBanner";
  el.innerHTML = banner("bad",
    "<b>Degraded mode — no classifier loaded.</b> " +
    "Detection, vetting features and transit fitting all still work, but " +
    "nothing on this page is classified: no transit / eclipse / blend call, " +
    "no confidence, no feature drivers. " +
    "Train one with <code>python scripts/train.py</code>, or on Linux check " +
    "that LightGBM found OpenMP (<code>apt-get install libgomp1</code>).");
  host.appendChild(el.firstElementChild);
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

/* ================================================================ analyse = */
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
  $("runBtn").textContent = "Running";

  const labels = { detrend: "Detrend and normalise", bls: "BLS period search",
                   features: "Vetting diagnostics", classify: "Classify and calibrate",
                   fit: body.run_mcmc ? "Transit fit · MCMC posterior" : "Transit fit" };
  $("result").innerHTML = `
    <div class="empty">
      <div class="stages">${STAGES.map((s) =>
        `<div class="stage" data-stage="${s}">${labels[s]}</div>`).join("")}</div>
      <p id="elapsed" style="font-family:var(--f-data);color:var(--muted-2)">0.0 s</p>
    </div>`;

  /* The API returns one result rather than streaming progress, so the stages
     advance on the order the pipeline actually runs them: each is marked
     started, and the last stays active until the response lands. Only the
     total elapsed time is measured and shown. */
  const t0 = performance.now();
  const tick = setInterval(() => {
    $("elapsed") && ($("elapsed").textContent =
      ((performance.now() - t0) / 1000).toFixed(1) + " s");
  }, 100);
  const marks = [[0, "detrend"], [500, "bls"], [1400, "features"],
                 [2000, "classify"], [2500, "fit"]];
  const timers = marks.map(([ms, s]) => setTimeout(() => setStage(s), ms));

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
    $("result").innerHTML = banner("bad", e.message);
    clearStages();
  } finally {
    clearInterval(tick);
    timers.forEach(clearTimeout);
    busy = false;
    $("runBtn").disabled = false;
    $("runBtn").textContent = "Analyse";
  }
}

/* ================================================================= render = */
function render(d) {
  const t = d.target;
  const head = `
    <div class="thead">
      <div class="nm">${t.name}</div>
      <div class="meta">
        ${t.source} · ${t.n_points.toLocaleString()} points ·
        ${fmt(t.baseline_days, 1)} d baseline · detrender: ${d.detrend_method} ·
        ${d.cached ? "cached" : fmt(d.elapsed_seconds, 1) + " s"}
        ${t.note ? " · " + t.note : ""}
      </div>
    </div>`;

  if (!d.detected) {
    $("result").innerHTML = head + `<div class="alerts">${banner("warn", d.message)}</div>` +
      (d.series ? `<div class="panel"><h3>Light curve</h3>
         <div class="chart" id="cRaw" style="height:300px"></div></div>` : "");
    if (d.series) drawRaw(d);
    setStage(null); clearStages();
    return;
  }

  const det = d.detection, cls = d.classification, fit = d.fit;
  const col = cls ? (CLASS_COLOR[cls.label] || T.brand) : T.brand;

  /* --- alerts, ordered by how much they should interrupt ----------------- */
  const alerts = [];

  /* The classifier and the fit disagree. This goes first deliberately: the
     failure mode it guards against is a reader seeing a confident label and
     stopping there. In an amber theme an ordinary red banner does not out-shout
     the chrome, so it gets the critical tier with a solid header strip. */
  if (d.caution_flag) {
    alerts.push(`<div class="banner critical bad"><span class="rail"></span>
      <span class="ico">${ICON.bad}</span>
      <span class="strip">${ICON.bad} Caution — treat this classification with suspicion</span>
      <span class="body">${d.caution_reason || "the fit does not describe the data"}</span>
    </div>`);
  }

  if (t.truth_label && cls) {
    const ok = t.truth_label === cls.label;
    alerts.push(banner(ok ? "ok" : "bad",
      `Injected truth: <b>${t.truth_label}</b> — prediction
       ${ok ? "matches" : `differs (<b>${cls.label}</b>)`}.`));
  } else if (t.published_period) {
    const ratio = det.period_days / t.published_period;
    const harm = [1, 2, 0.5, 3, 1 / 3].find((m) => Math.abs(ratio / m - 1) < 0.02);
    const tag = harm === 1 ? "matches exactly"
              : harm ? `is the ${harm === 0.5 ? "½" : harm}× harmonic of`
              : "does NOT match";
    alerts.push(banner(harm === 1 ? "ok" : harm ? "warn" : "bad",
      `Recovered period <b>${fmt(det.period_days, 5)} d</b> ${tag}
       the published <b>${t.published_period} d</b>.`));
  }

  /* The retained detection came from the fallback detrender, so its SDE is the
     better of two trials rather than a single measurement. */
  if (d.detrend_is_fallback && d.sde_corrected != null) {
    alerts.push(banner("warn",
      `Detected by the fallback <b>${d.detrend_method}</b> detrender, not the
       primary one. Across ${d.n_detrend_trials} trials the SDE of
       <b>${fmt(det.sde, 2)}</b> is worth <b>${fmt(d.sde_corrected, 2)}</b> after
       a look-elsewhere discount.`));
  }

  $("result").innerHTML = head +
    (alerts.length ? `<div class="alerts">${alerts.join("")}</div>` : "") + `
    <div class="metrics">
      <div class="card hero" style="--cls:${col}">
        <div>
          <span class="k">Classification</span>
          <div class="v">${cls ? cls.label : "—"}</div>
          <div class="s">${cls
            ? `calibrated · ${(cls.confidence * 100).toFixed(1)}% confidence`
            : "no classifier loaded"}</div>
        </div>
        ${cls ? confRing(cls.confidence) : ""}
      </div>
      <div class="card"><span class="k">Orbital period</span>
        <div class="v">${fmt(det.period_days, 4)}<span class="u"> d</span></div>
        <div class="s">${det.n_transits} transits observed</div></div>
      <div class="card"><span class="k">Transit depth</span>
        <div class="v">${fmt(det.depth_ppm, 0)}<span class="u"> ppm</span></div>
        <div class="s">duration ${fmt(det.duration_hours, 2)} h</div></div>
      <div class="card"><span class="k">Significance</span>
        <div class="v">${fmt(det.sde, 1)}<span class="u"> SDE</span></div>
        <div class="s">S/N ${fmt(det.snr, 0)}${
          d.detrend_is_fallback && d.sde_corrected != null
            ? ` · ${fmt(d.sde_corrected, 1)} corrected` : ""}</div></div>
    </div>

    ${cls ? classPanel(cls) : ""}

    <div class="panel"><h3>Light curve <em>raw with fitted trend, then detrended</em></h3>
      <div class="chart" id="cRaw" style="height:220px"></div>
      <div class="chart" id="cDet" style="height:180px"></div></div>

    <div class="row2">
      <div class="panel"><h3>BLS periodogram</h3>
        <div class="chart" id="cPg" style="height:230px"></div>
        <div class="cap">Dashed mark at the recovered period; dotted at 2P and P/2</div></div>
      <div class="panel"><h3>Phase-folded</h3>
        <div class="chart" id="cFold" style="height:230px"></div>
        <div class="cap">Binned points ±1σ with the fitted model</div></div>
    </div>

    <div class="panel"><h3>Vetting diagnostics</h3>
      <div class="row2" style="grid-template-columns:1fr 1fr">
        <div><div class="chart" id="cOE" style="height:220px"></div>
          <div class="cap">Odd vs even transits — they separate for an eclipsing binary</div></div>
        <div><div class="chart" id="c2P" style="height:220px"></div>
          <div class="cap">Folded at 2×P — a planet shows two equal events,
            an EB a shallower secondary</div></div>
      </div></div>

    ${fit ? fitPanel(fit, t) : ""}

    <div class="panel"><details><summary>Vetting feature vector · ${
      d.features ? Object.keys(d.features).length : 0} values</summary>
      <div class="features">
        ${Object.entries(d.features || {}).map(([k, v]) =>
          `<div class="frow"><span>${k}</span><span>${fmt(v, 4)}</span></div>`).join("")}
      </div></details></div>`;

  drawRaw(d); drawDet(d); drawPg(d); drawFold(d); drawOE(d); draw2P(d);
  setStage(null);
  $("stepper").querySelectorAll("b").forEach((b) => b.className = "done");
}

function confRing(c) {
  const r = 23, circ = 2 * Math.PI * r;
  return `<div class="conf">
    <svg width="58" height="58" viewBox="0 0 58 58" aria-hidden="true">
      <circle cx="29" cy="29" r="${r}" fill="none" stroke="var(--line)" stroke-width="5"/>
      <circle cx="29" cy="29" r="${r}" fill="none" stroke="var(--cls)" stroke-width="5"
        stroke-dasharray="${(circ * c).toFixed(1)} ${circ.toFixed(1)}"
        transform="rotate(-90 29 29)"/>
    </svg><b>${Math.round(c * 100)}%</b></div>`;
}

/* Five bars do not justify a Plotly instance: native markup is crisper, reads
   the class tokens straight from the stylesheet, and animates in. */
function classPanel(cls) {
  const rows = Object.entries(cls.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v], i) => `
      <div class="prob${i === 0 ? " lead" : ""}" style="--cls:${CLASS_COLOR[k] || T.brand}">
        <span class="lab">${k}</span>
        <span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%"></span></span>
        <span class="val">${(v * 100).toFixed(1)}%</span>
      </div>`).join("");

  /* SHAP contributions are signed, so they read spatially: a bar right of the
     axis pushed the model toward this label, left of it away. */
  const drivers = cls.drivers || [];
  const scale = Math.max(0.01, ...drivers.map(([, v]) => Math.abs(v)));
  const shap = drivers.length ? `
    <div class="shap">
      <div class="cap label" style="text-align:left">Evidence · SHAP contribution</div>
      ${drivers.map(([k, v]) => {
        const w = (Math.abs(v) / scale) * 50;
        const side = v >= 0
          ? `left:50%;width:${w}%;background:${CLASS_COLOR[cls.label] || T.brand}`
          : `right:50%;width:${w}%;background:${T.muted}`;
        return `<div class="sh"><span class="f" title="${k}">${k}</span>
          <span class="ax"><span class="b" style="${side}"></span></span>
          <span class="n">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></div>`;
      }).join("")}
    </div>` : "";

  return `<div class="panel">
    <h3>Classification <em>calibrated probabilities · isotonic</em></h3>
    <div class="probs">${rows}</div>
    <p class="blurb">${CLASS_BLURB[cls.label] || ""}</p>
    ${shap}</div>`;
}

function fitPanel(f, t) {
  const pm = (v, e, n) => e === null || e === undefined
    ? fmt(v, n) : `${fmt(v, n)} <span class="err">± ${fmt(e, n)}</span>`;
  const cmp = (v, p, n) => p ? `${fmt(p, n)} <span class="err">(${
    ((v / p - 1) * 100 >= 0 ? "+" : "")}${((v / p - 1) * 100).toFixed(1)}%)</span>` : "—";

  /* Five of seven rows have no published counterpart on a simulated target;
     an all-em-dash column is wasted width, so it only appears when populated. */
  const hasPub = !!(t.published_period || t.published_rp_rs);

  const rows = [
    ["Orbital period", pm(f.period_days, f.period_err, 5), "days",
      t.published_period ? fmt(t.published_period, 5) : "—"],
    ["Transit depth (observed)",
      pm(f.depth_observed_ppm, f.depth_observed_err_ppm, 0), "ppm", "—"],
    ["Transit depth (geometric, Rp²/R*²)",
      pm(f.depth_geometric_ppm, f.depth_geometric_err_ppm, 0), "ppm", "—"],
    ["Transit duration", pm(f.duration_hours, f.duration_err_hours, 3), "hours", "—"],
    ["Rp / R*", pm(f.rp_over_rs, f.rp_over_rs_err, 5), "—",
      cmp(f.rp_over_rs, t.published_rp_rs, 5)],
    ["Impact parameter", fmt(f.impact_param, 3), "—", "—"],
    ["Reduced χ²", `<span class="${f.reliable ? "chi-ok" : "chi-bad"}">${
      fmt(f.chi2_reduced, 2)}</span>`, "—", "—"],
  ];

  return `<div class="panel">
    <h3>Fitted transit model <em>Mandel–Agol · MCMC, 32 walkers</em></h3>
    ${f.warning ? banner("warn", f.warning) : ""}
    <div class="tablewrap"><table>
      <thead><tr><th>Parameter</th><th class="r">Value ± 1σ</th><th>Unit</th>
        ${hasPub ? '<th class="r">Published</th>' : ""}</tr></thead>
      <tbody>${rows.map(([p, v, u, pub]) => `<tr>
        <td>${p}</td><td class="num">${v}</td><td class="unit">${u}</td>
        ${hasPub ? `<td class="num">${pub}</td>` : ""}</tr>`).join("")}</tbody>
    </table></div>
    <p class="cap" style="text-align:left;margin-top:var(--s3)">
      Uncertainties are the standard deviation of the MCMC posterior. The observed
      depth exceeds the geometric depth because limb darkening makes the stellar
      disc brighter at its centre. A reduced χ² far from 1 means the model does
      not describe the data — read the classification with suspicion.</p></div>`;
}

/* ================================================================= charts = */
const sc = (s, o = {}) => ({
  x: s.x, y: s.y, type: "scattergl", mode: "markers",
  marker: { size: 2, color: T.muted2 }, hoverinfo: "skip", ...o,
});

function drawRaw(d) {
  const s = d.series, tr = [sc(s.raw)];
  if (s.trend.x.length)
    tr.push({ x: s.trend.x, y: s.trend.y, type: "scattergl", mode: "lines",
              line: { color: T.eclipse, width: 1.4 }, hoverinfo: "skip" });
  Plotly.newPlot("cRaw", tr,
    layout({ yaxis: { ...LAYOUT.yaxis, title: "raw flux" },
             xaxis: { ...LAYOUT.xaxis, title: "" },
             margin: { l: 56, r: 14, t: 6, b: 20 } }), CONFIG);
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
          line: { color: T.brand, width: 1, dash: "dot" } });
      }
    }
  }
  Plotly.newPlot("cDet", [sc(s.detrended, { marker: { size: 2, color: T.muted } })],
    layout({ shapes, yaxis: { ...LAYOUT.yaxis, title: "detrended" },
             xaxis: { ...LAYOUT.xaxis, title: "time (days)" },
             margin: { l: 56, r: 14, t: 6, b: 36 } }), CONFIG);
}

function drawPg(d) {
  const s = d.series, P = d.detection.period_days;
  const shapes = [{ type: "line", x0: Math.log10(P), x1: Math.log10(P), y0: 0, y1: 1,
                    yref: "paper", line: { color: T.eclipse, width: 1.4, dash: "dash" } }];
  [2, 0.5].forEach((m) => {
    const p = P * m;
    if (p >= Math.min(...s.periodogram.x) && p <= Math.max(...s.periodogram.x))
      shapes.push({ type: "line", x0: Math.log10(p), x1: Math.log10(p), y0: 0, y1: 1,
                    yref: "paper", line: { color: T.blend, width: 1, dash: "dot" } });
  });
  Plotly.newPlot("cPg",
    [{ x: s.periodogram.x, y: s.periodogram.y, type: "scattergl", mode: "lines",
       line: { color: T.brand, width: 1 }, hovertemplate: "P=%{x:.4f} d<extra></extra>" }],
    layout({ shapes, xaxis: { ...LAYOUT.xaxis, title: "period (days)", type: "log" },
             yaxis: { ...LAYOUT.yaxis, title: "BLS power" } }), CONFIG);
}

function drawFold(d) {
  const s = d.series, det = d.detection;
  const half = Math.min(0.5, 3 * (det.duration_hours / 24) / det.period_days);
  const tr = [
    sc(s.fold, { marker: { size: 2, color: T.muted2 } }),
    { x: s.fold_binned.x, y: s.fold_binned.y, type: "scatter", mode: "markers",
      marker: { size: 5, color: T.text },
      error_y: { type: "data", array: s.fold_binned.e, width: 0, color: T.muted },
      hovertemplate: "phase %{x:.4f}<br>%{y:.6f}<extra></extra>" },
  ];
  if (s.model)
    tr.push({ x: s.model.x, y: s.model.y, type: "scatter", mode: "lines",
              line: { color: T.eclipse, width: 2 }, hoverinfo: "skip" });
  Plotly.newPlot("cFold", tr,
    layout({ xaxis: { ...LAYOUT.xaxis, title: "phase", range: [-half, half] },
             yaxis: { ...LAYOUT.yaxis, title: "normalised flux" } }), CONFIG);
}

function drawOE(d) {
  const s = d.series, tr = [];
  if (s.even) tr.push({ x: s.even.x, y: s.even.y, type: "scatter", mode: "lines+markers",
                        line: { color: T.brand, width: 1.6 }, marker: { size: 3 }, name: "even" });
  if (s.odd) tr.push({ x: s.odd.x, y: s.odd.y, type: "scatter", mode: "lines+markers",
                       line: { color: T.eclipse, width: 1.6 }, marker: { size: 3 }, name: "odd" });
  Plotly.newPlot("cOE", tr,
    layout({ showlegend: true,
             legend: { orientation: "h", y: 1.18, x: 0, font: { size: 10 } },
             xaxis: { ...LAYOUT.xaxis, title: "hours from mid-transit" },
             yaxis: { ...LAYOUT.yaxis, title: "flux" } }), CONFIG);
}

function draw2P(d) {
  const s = d.series;
  const shapes = [0, 0.5, -0.5].map((x) => ({
    type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper",
    line: { color: x === 0 ? T.brand : T.eclipse, width: 1, dash: "dot" } }));
  Plotly.newPlot("c2P", [sc(s.fold_2p, { marker: { size: 3, color: T.muted } })],
    layout({ shapes, xaxis: { ...LAYOUT.xaxis, title: "phase (period = 2P)" },
             yaxis: { ...LAYOUT.yaxis, title: "flux" } }), CONFIG);
}

/* ================================================================= events = */
$("runBtn").addEventListener("click", () => run());
$("ticBtn").addEventListener("click", () => {
  const v = $("ticInput").value.trim();
  if (v) {
    document.querySelectorAll(".target").forEach((x) => x.classList.remove("active"));
    selection = null;
    document.body.classList.remove("nav-open");
    run({ tic: v });
  }
});
$("ticInput").addEventListener("keydown", (e) => { if (e.key === "Enter") $("ticBtn").click(); });

$("menuBtn").addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  $("menuBtn").setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.body.classList.remove("nav-open");
});

boot();
