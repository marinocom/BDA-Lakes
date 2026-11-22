// Posterior predictive for the break-up day of a lake, from M4 posterior draws.
// Works in the browser (window.IcePredict) and in Node (module.exports) for testing.
(function (root) {
  "use strict";

  // Small seeded random generator so the same lake gives the same answer.
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(u) {
    let a = 0, b = 0;
    while (a === 0) a = u();
    b = u();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  }
  // Marsaglia-Tsang gamma(shape, 1)
  function gamma(shape, u) {
    if (shape < 1) return gamma(shape + 1, u) * Math.pow(u(), 1 / shape);
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = normal(u); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const w = u();
      if (w < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }
  function studentT(nu, u) {
    return normal(u) / Math.sqrt(gamma(nu / 2, u) / (nu / 2));
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371, r = Math.PI / 180;
    const dlat = (lat2 - lat1) * r, dlon = (lon2 - lon1) * r;
    const a = Math.sin(dlat / 2) ** 2 + Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(dlon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(a));
  }

  // Inverse-distance guess of the altitude from the observation sites (only a starting value).
  function guessAltitude(P, lat, lon) {
    let num = 0, den = 0;
    for (const s of P.sites) {
      const dkm = Math.max(haversineKm(lat, lon, s.lat, s.lon), 1);
      const w = 1 / (dkm * dkm);
      num += w * s.alt; den += w;
    }
    return num / den;
  }

  function nearestSite(P, lat, lon) {
    let best = null, bestD = Infinity;
    for (const s of P.sites) {
      const dkm = haversineKm(lat, lon, s.lat, s.lon);
      if (dkm < bestD) { bestD = dkm; best = s; }
    }
    return { site: best, km: bestD };
  }

  // Predictive draws of the break-up day of year.
  // opts: {lat, lon, alt, year, site (optional observed site object), perDraw, seed}
  function predict(P, opts) {
    const D = P.draws, n = P.n_draws;
    const perDraw = opts.perDraw || 8;
    const u = rng(opts.seed || 12345);
    const lc = opts.lat - P.lat_center;
    const oc = opts.lon - P.lon_center;
    const ac = (opts.alt - P.alt_center) / P.alt_scale;
    const x = (opts.year - 1990) / 10;
    const p = Math.max(opts.year - P.break_year, 0) / 10;
    const out = [], trend = [];
    for (let i = 0; i < n; i++) {
      let mu = D["Intercept"][i] + D["lat_c"][i] * lc + D["lon_c"][i] * oc + D["alt_c"][i] * ac
        + (D["decade_c"][i] + D["lat_c:decade_c"][i] * lc) * x + D["post1980"][i] * p;
      trend.push(mu);
      for (let k = 0; k < perDraw; k++) {
        let lake;
        if (opts.site) {
          lake = opts.site.a[i] + opts.site.b[i] * x;     // the observed lake's own effects
        } else {
          lake = normal(u) * D["1|site_sigma"][i] + normal(u) * D["decade_c|site_sigma"][i] * x;
        }
        const weather = normal(u) * D["1|season_f_sigma"][i] + normal(u) * D["1|season_region_sigma"][i];
        out.push(mu + lake + weather + D["sigma"][i] * studentT(D["nu"][i], u));
      }
    }
    return { draws: out, trend: trend };
  }

  function quantile(arr, q) {
    const a = arr.slice().sort((x, y) => x - y);
    const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }

  function doyToDate(doy, year) {
    const d = new Date(Date.UTC(year, 0, 1));
    d.setUTCDate(d.getUTCDate() + Math.round(doy) - 1);
    return d;
  }

  const api = { predict, quantile, doyToDate, guessAltitude, nearestSite, haversineKm, rng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.IcePredict = api;
})(typeof window !== "undefined" ? window : this);
