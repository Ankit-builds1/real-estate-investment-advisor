// In-browser inference: mirrors src/features.py + the sklearn ColumnTransformer + XGBoost trees.
const AMENITIES = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"];
const TRANSPORT = { Low: 1, Medium: 2, High: 3 };

export function addFeatures(p) {
  const f = { ...p };
  f.Parking_Flag = p.Parking_Space === "Yes" ? 1 : 0;
  f.Security_Flag = p.Security === "Yes" ? 1 : 0;
  const amen = p.Amenities || "";
  AMENITIES.forEach((a) => (f[`Has_${a}`] = amen.includes(a) ? 1 : 0));
  f.Amenity_Count = AMENITIES.reduce((s, a) => s + f[`Has_${a}`], 0);
  f.School_Density_Score = p.Nearby_Schools / 10;
  f.Hospital_Density_Score = p.Nearby_Hospitals / 10;
  const t = TRANSPORT[p.Public_Transport_Accessibility] ?? 2;
  f.Infrastructure_Score = Math.round(((f.School_Density_Score + f.Hospital_Density_Score + (t - 1) / 2) / 3) * 1000) / 1000;
  f.Floor_Ratio = Math.min(1, Math.max(0, p.Floor_No / (p.Total_Floors || 1)));
  return f;
}

// Same growth formula used to create the regression target
export function growthRate(f) {
  let r = 0.08 + (f.Infrastructure_Score - 0.5) * 0.04;
  if (f.Availability_Status === "Under_Construction") r += 0.01;
  if (f.Age_of_Property > 25) r -= 0.01;
  r += f.Amenity_Count * 0.002;
  if (f.Property_Type === "Villa") r += 0.005;
  return r;
}

function transform(model, f) {
  const x = [];
  model.numeric.forEach((c, i) => x.push(Math.fround((Number(f[c]) - model.mean[i]) / model.scale[i])));
  model.categorical.forEach((c, i) => model.categories[i].forEach((v) => x.push(f[c] === v ? 1 : 0)));
  return x;
}

function walk(node, x) {
  while (Array.isArray(node)) {
    const [feat, thr, yes, no, miss] = node;
    const v = x[feat];
    // Pipeline output is sparse: XGBoost treats absent (zero) entries as missing
    if (v === 0) node = miss === "y" ? yes : no;
    else node = Math.fround(v) < Math.fround(thr) ? yes : no;
  }
  return node;
}

function margin(model, x) {
  let s = 0;
  for (const t of model.trees) s += walk(t, x);
  return s;
}

export function predictProba(model, f) {
  const b = model.base_score;
  const m = margin(model, transform(model, f)) + Math.log(b / (1 - b));
  return 1 / (1 + Math.exp(-m));
}

export function predictPrice(model, f) {
  return margin(model, transform(model, f)) + model.base_score;
}
