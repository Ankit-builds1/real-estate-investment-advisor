import { addFeatures, growthRate, predictPrice, predictProba } from "./model.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmtL = (v) => `₹${v.toLocaleString("en-IN", { maximumFractionDigits: 1, minimumFractionDigits: 1 })} L`;
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const label = (s) => s.replaceAll("_", " ");

const [clf, reg, market] = await Promise.all(
  ["classifier", "regressor", "market"].map((n) => fetch(`data/${n}.json`).then((r) => r.json()))
);

/* ---------------- hero stats ---------------- */
const best = market.metrics;
$("#rowsLabel").textContent = market.rows.toLocaleString("en-IN");
$("#heroStats").innerHTML = [
  [market.rows.toLocaleString("en-IN"), "properties"],
  [market.cities, "cities"],
  [pct(best.classification.XGBoost.accuracy, 2), "verdict accuracy"],
  [best.regression.XGBoost.r2.toFixed(4), "price model R²"],
].map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");

/* ---------------- form ---------------- */
const form = $("#form");
const stateSel = $("#state"), citySel = $("#city");
Object.keys(market.city_by_state).sort().forEach((s) => stateSel.add(new Option(s, s, s === "Odisha", s === "Odisha")));
function fillCities() {
  citySel.innerHTML = "";
  market.city_by_state[stateSel.value].forEach((c) => citySel.add(new Option(c, c)));
}
fillCities();
stateSel.addEventListener("change", () => { fillCities(); update(); });

$$(".seg").forEach((seg) =>
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
    update();
  })
);
function paintRange(r) {
  r.style.setProperty("--p", `${((r.value - r.min) / (r.max - r.min)) * 100}%`);
  const o = $(`output[data-for="${r.name}"]`);
  if (o) o.textContent = r.value;
}
$$("input[type=range]").forEach(paintRange);
form.addEventListener("input", (e) => {
  if (e.target.type === "range") paintRange(e.target);
  update();
});

function readProperty() {
  const fd = new FormData(form);
  const v = (n) => fd.get(n);
  const seg = (n) => $(`.seg[data-name="${n}"] .on`).dataset.v;
  const price = Math.max(1, Number(v("Price_in_Lakhs")) || 1);
  const size = Math.max(100, Number(v("Size_in_SqFt")) || 100);
  const total = Number(v("Total_Floors"));
  const year = Number(v("Year_Built"));
  return {
    State: v("State"), City: v("City"), Property_Type: seg("Property_Type"), BHK: Number(v("BHK")),
    Size_in_SqFt: size, Price_in_Lakhs: price, Price_per_SqFt: (price * 1e5) / size, Year_Built: year,
    Furnished_Status: v("Furnished_Status"), Floor_No: Math.min(Number(v("Floor_No")), total), Total_Floors: total,
    Age_of_Property: 2025 - year, Nearby_Schools: Number(v("Nearby_Schools")), Nearby_Hospitals: Number(v("Nearby_Hospitals")),
    Public_Transport_Accessibility: seg("Public_Transport_Accessibility"),
    Parking_Space: fd.get("Parking_Space") ? "Yes" : "No", Security: fd.get("Security") ? "Yes" : "No",
    Amenities: $$("#amenities input:checked").map((i) => i.value).join(", "),
    Facing: v("Facing"), Owner_Type: v("Owner_Type"), Availability_Status: seg("Availability_Status"),
  };
}

/* ---------------- deed ---------------- */
let lastGood = null, deedNo = 1;
function update() {
  const f = addFeatures(readProperty());
  const p = predictProba(clf, f);
  const future = predictPrice(reg, f);
  const good = p >= 0.5;
  const conf = good ? p : 1 - p;
  const cagr = Math.pow(future / f.Price_in_Lakhs, 1 / 5) - 1;

  const stamp = $("#stamp");
  stamp.classList.toggle("bad", !good);
  $("#stampText").innerHTML = good ? "GOOD" : "BAD";
  if (lastGood !== null && lastGood !== good) {
    stamp.classList.remove("thump"); void stamp.offsetWidth; stamp.classList.add("thump");
    deedNo++;
  }
  lastGood = good;
  $("#deedNo").textContent = `No. ${String(deedNo).padStart(4, "0")}`;
  $("#verdict").textContent = good ? "Good Investment" : "Not a Good Investment";
  $("#confidence").textContent = pct(conf);
  const fill = $("#meterFill");
  fill.style.width = pct(conf);
  fill.style.background = good ? "var(--good)" : "var(--bad)";

  $("#nowPrice").textContent = fmtL(f.Price_in_Lakhs);
  $("#futurePrice").textContent = fmtL(future);
  $("#gain").textContent = `+${fmtL(future - f.Price_in_Lakhs)} gain · ${pct(cagr, 2)} per year`;
  drawGrowth(f.Price_in_Lakhs, cagr);

  const med = market.city_median_ppsf[f.City];
  const checks = [
    [f.Price_per_SqFt <= med, `Price/sq ft vs ${f.City} median`, `₹${Math.round(f.Price_per_SqFt).toLocaleString("en-IN")} / ₹${Math.round(med).toLocaleString("en-IN")}`],
    [f.BHK >= 3, "3 or more bedrooms", `${f.BHK} BHK`],
    [f.Availability_Status === "Ready_to_Move", "Ready to move in", label(f.Availability_Status)],
    [f.Infrastructure_Score >= 0.5, "Strong infrastructure", f.Infrastructure_Score.toFixed(2)],
    [growthRate(f) >= 0.08, "Growth at or above 8% a year", pct(growthRate(f), 2)],
  ];
  $("#checks").innerHTML = checks.map(([ok, t, v]) =>
    `<div class="check ${ok ? "ok" : ""}"><i>${ok ? "✓" : "–"}</i><span>${t}</span><b>${v}</b></div>`).join("");
}

function drawGrowth(p0, r) {
  const svg = $("#growthChart");
  const W = 400, H = 140, pad = 18;
  const pts = [0, 1, 2, 3, 4, 5].map((y) => p0 * Math.pow(1 + r, y));
  const lo = p0 * 0.97, hi = pts[5] * 1.02;
  const x = (i) => pad + (i / 5) * (W - pad * 2);
  const y = (v) => H - 22 - ((v - lo) / (hi - lo)) * (H - 40);
  const line = pts.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const css = getComputedStyle(document.documentElement);
  const acc = css.getPropertyValue("--accent").trim(), ink = css.getPropertyValue("--ink-2").trim(), rule = css.getPropertyValue("--rule").trim();
  svg.innerHTML = `
    <defs><linearGradient id="gf" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${acc}" stop-opacity=".28"/><stop offset="1" stop-color="${acc}" stop-opacity="0"/></linearGradient></defs>
    ${[0, 1, 2, 3, 4, 5].map((i) => `<line x1="${x(i)}" x2="${x(i)}" y1="10" y2="${H - 22}" stroke="${rule}" stroke-dasharray="2 4"/>`).join("")}
    <path d="${line}L${x(5)},${H - 22}L${x(0)},${H - 22}Z" fill="url(#gf)"/>
    <path d="${line}" fill="none" stroke="${acc}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
    ${pts.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="${i === 5 ? 4.5 : 2.5}" fill="${i === 5 ? acc : "var(--card)"}" stroke="${acc}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`).join("")}
    ${[0, 1, 2, 3, 4, 5].map((i) => `<text x="${x(i)}" y="${H - 4}" text-anchor="middle" font-size="10" fill="${ink}">${i === 0 ? "Now" : "Y" + i}</text>`).join("")}`;
}

/* ---------------- market charts ---------------- */
function bars(el, entries, { fmt = (v) => v, max, min = 0, hot = () => false } = {}) {
  const mx = max ?? Math.max(...entries.map(([, v]) => v));
  el.innerHTML = entries.map(([k, v]) => k === "—" ? `<div class="sep"></div>` :
    `<div class="bar"><span title="${k}">${k}</span><div class="track"><i class="${hot(k, v) ? "hot" : ""}" style="width:${Math.max(2, ((v - min) / (mx - min)) * 100)}%"></i></div><b>${fmt(v)}</b></div>`).join("");
}
const ord = ["Low", "Medium", "High"];
bars($("#transportChart"), ord.map((k) => [k, market.transport_good[k]]), { fmt: (v) => pct(v), max: 1, hot: (k) => k === "High" });
bars($("#statusChart"), Object.entries(market.status_good).map(([k, v]) => [label(k), v]), { fmt: (v) => pct(v), max: 1, hot: (k) => k.startsWith("Ready") });
$("#bhkChart").innerHTML = Object.entries(market.bhk_good).map(([k, v]) =>
  `<div class="col"><b>${pct(v, 0)}</b><i style="height:${v * 100}%"></i><span>${k} BHK</span></div>`).join("");

const st = Object.entries(market.state_ppsf);
const stMin = Math.min(...st.map(([, v]) => v));
bars($("#stateChart"), st, { fmt: (v) => `₹${Math.round(v / 100) / 10}k`, min: stMin * 0.98, hot: (_, v) => v === st[0][1] });

const cg = Object.entries(market.city_good);
$("#cityCount").textContent = cg.length;
const cgMin = Math.min(...cg.map(([, v]) => v));
bars($("#cityChart"), [...cg.slice(0, 5), ["—", 0], ...cg.slice(-5)], { fmt: (v) => pct(v), min: cgMin * 0.97, hot: (k) => cg.slice(0, 5).some(([c]) => c === k) });

(function hist() {
  const h = market.growth_hist, e = market.growth_edges, mx = Math.max(...h);
  const w = 400 / h.length;
  $("#histChart").innerHTML = h.map((v, i) => `<rect x="${i * w + 1}" width="${w - 2}" y="${120 - (v / mx) * 115}" height="${(v / mx) * 115}" fill="${(e[i] + e[i + 1]) / 2 >= 8 ? "var(--accent)" : "var(--bar)"}" opacity=".9"/>`).join("");
  $("#histMin").textContent = `${e[0].toFixed(1)}%`;
  $("#histMax").textContent = `${e[e.length - 1].toFixed(1)}%`;
})();

/* ---------------- models ---------------- */
function table(el, data, cols, bestName, fmts) {
  el.innerHTML = `<thead><tr><th>Model</th>${cols.map((c) => `<th>${c.toUpperCase()}</th>`).join("")}</tr></thead><tbody>` +
    Object.entries(data).map(([m, r]) => `<tr class="${m === bestName ? "best" : ""}"><td>${m.replace(/([a-z])([A-Z])/g, "$1 $2")}</td>${cols.map((c) => `<td>${fmts(c, r[c])}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
table($("#clfTable"), best.classification, ["accuracy", "f1", "roc_auc"], best.best_classifier, (_, v) => v.toFixed(4));
table($("#regTable"), best.regression, ["rmse", "mae", "r2"], best.best_regressor, (c, v) => (c === "r2" ? v.toFixed(4) : v.toFixed(2)));
const imp = (o) => Object.entries(o).slice(0, 8).map(([k, v]) => [label(k), v]);
bars($("#clfImp"), imp(market.importance.clf), { fmt: (v) => v.toFixed(3) });
bars($("#regImp"), imp(market.importance.reg), { fmt: (v) => v.toFixed(3) });

/* ---------------- animate charts into view ---------------- */
const io = new IntersectionObserver((ents) => ents.forEach((en) => {
  if (!en.isIntersecting) return;
  $$("[data-w]", en.target).forEach((i) => (i.style.width = i.dataset.w + "%"));
  $$("[data-h]", en.target).forEach((i) => (i.style.height = i.dataset.h + "%"));
  io.unobserve(en.target);
}), { threshold: 0.2 });
$$(".panel").forEach((p) => io.observe(p));

update();
