// Page interactions. No dependencies besides KaTeX for the equations.

const clamp = (x, a = 0, b = 1) => Math.min(Math.max(x, a), b);
const smooth = (x) => x * x * (3 - 2 * x);
const pad3 = (i) => String(i).padStart(3, "0");
const NS = "http://www.w3.org/2000/svg";
const svg = (parent, name, attrs = {}, text) => {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.append(e);
  return e;
};
const loadJSON = async (url) => (await fetch(url)).json();
const preload = (urls) => urls.map((u) => { const im = new Image(); im.src = u; return im; });

// Run an animation until the viewer takes over.
function autoplay(step, ms) {
  const id = setInterval(step, ms);
  return () => clearInterval(id);
}

// Videos play only while on screen.
const visible = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const vids = e.target.tagName === "VIDEO" ? [e.target] : e.target.querySelectorAll("video");
    for (const v of vids) e.isIntersecting ? v.play().catch(() => {}) : v.pause();
  }
}, { threshold: 0.1 });

/* Hero: tile labels follow the wall video; the veil and title lift as the page scrolls. */
async function initHero() {
  const hero = document.getElementById("hero");
  const wall = hero.querySelector(".wall");
  const video = wall.querySelector("video");
  const veil = wall.querySelector(".veil");
  const title = wall.querySelector(".hero-title");
  const labels = await loadJSON("media/wall.json");
  // Tile geometry of the composed wall (wall_filter.py), as fractions of its size.
  const TW = 620, TH = 349, GAP = 14, W = 3 * TW + 2 * GAP, H = 2 * TH + GAP;
  const box = wall.querySelector(".wall-labels");
  const tags = labels.map((_, k) => {
    const b = document.createElement("b");
    b.style.left = `${(((k % 3) * (TW + GAP) + TW) / W) * 100}%`;
    b.style.top = `${((Math.floor(k / 3) * (TH + GAP) + TH) / H) * 100}%`;
    box.append(b);
    return b;
  });
  const t0 = performance.now();
  const frame = () => {
    const i = Math.floor(video.currentTime * 30) % labels[0].length;
    tags.forEach((b, k) => { b.textContent = labels[k][i]; });
    const intro = smooth(clamp((performance.now() - t0 - 800) / 900));
    const lift = clamp(window.scrollY / (wall.offsetHeight * 0.6));
    veil.style.opacity = (0.72 * intro * (1 - lift)).toFixed(3);
    title.style.opacity = (1 - lift).toFixed(3);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/* Before/after slider over two frame-locked videos. */
function initCompare(box) {
  const [top, base] = box.querySelectorAll("video");
  const set = (clientX) => {
    const r = box.getBoundingClientRect();
    box.style.setProperty("--x", `${(clamp((clientX - r.left) / r.width) * 100).toFixed(2)}%`);
  };
  box.addEventListener("pointerdown", (e) => { box.setPointerCapture(e.pointerId); set(e.clientX); });
  box.addEventListener("pointermove", (e) => { if (box.hasPointerCapture(e.pointerId)) set(e.clientX); });
  const sync = () => {
    if (Math.abs(top.currentTime - base.currentTime) > 1 / 30) top.currentTime = base.currentTime;
    requestAnimationFrame(sync);
  };
  requestAnimationFrame(sync);
  visible.observe(box);
}

/* Field of view against centre resolution: one slider, native in the middle. */
async function initTradeoff(root) {
  const count = await loadJSON("media/tradeoff/count.json");
  const ppd = await loadJSON("media/tradeoff/ppd.json");
  const seq = (dir, n) => preload([...Array(n).keys()].map((i) => `media/tradeoff/${dir}/${pad3(i)}.jpg`));
  const plain = { fov: seq("fov", count.fov), centre: seq("centre", count.centre) };
  const rings = { fov: seq("ring_fov", count.fov), centre: seq("ring_centre", count.centre) };
  const img = root.querySelector(".to-view .media");
  const state = root.querySelector(".to-state");
  const field = root.querySelector(".to-field");
  const focal = root.querySelector(".to-focal");
  const range = root.querySelector("input[type=range]");
  const toggle = root.querySelector(".to-rings");
  const legend = root.querySelector(".bands");
  const chart = drawPpdChart(root.querySelector(".to-chart"), ppd);
  let showRings = true;

  const show = (v) => {
    const u = Math.abs(v);
    const frames = (showRings ? rings : plain)[v < 0 ? "fov" : "centre"];
    img.src = frames[Math.round(u * (frames.length - 1))].src;
    const s = smooth(clamp((u - 0.15) / 0.7));  // the numbers follow the morph's easing
    field.textContent = `${Math.round(170 + ((v < 0 ? 150 : 95) - 170) * s)}°`;
    focal.textContent = `${Math.round(364 + ((v < 0 ? 107 : 364) - 364) * s)} px`;
    const side = u < 0.5 ? "kb" : v < 0 ? "p107" : "p364";
    state.textContent = { kb: "Native fisheye", p107: "Undistorted, 150°", p364: "Undistorted, 95°" }[side];
    for (const k in chart.curves) chart.curves[k].classList.toggle("on", k === side);
  };
  const setRings = (on) => {
    showRings = on;
    toggle.setAttribute("aria-pressed", String(on));
    toggle.textContent = on ? "Hide incidence bands" : "Show incidence bands";
    legend.classList.toggle("on", on);
    chart.bands.style.opacity = on ? 1 : 0;
    show(+range.value);
  };
  toggle.addEventListener("click", () => setRings(!showRings));
  setRings(true);
  let t = 0;
  const stop = autoplay(() => { t += 0.04; range.value = clamp(Math.sin(t) * 1.1, -1, 1); show(+range.value); }, 50);
  range.addEventListener("input", () => { stop(); show(+range.value); });
}

function drawPpdChart(el, d) {
  const W = 480, H = 300, L = 46, R = 12, T = 26, B = 36;
  const xs = (deg) => L + ((W - L - R) * deg) / 85;
  const ys = (v) => T + (H - T - B) * (1 - (Math.log(v) - Math.log(1.2)) / (Math.log(40) - Math.log(1.2)));
  const bands = svg(el, "g", { style: "opacity:0;transition:opacity .2s" });
  [[0, 20, "#0072B2"], [20, 40, "#56B4E9"], [40, 60, "#009E73"], [60, 85, "#E69F00"]].forEach(([lo, hi, c]) => {
    svg(bands, "rect", { x: xs(lo), y: T, width: xs(hi) - xs(lo), height: H - T - B, fill: c, "fill-opacity": 0.16 });
  });
  for (const v of [2, 5, 10, 20, 40]) {
    svg(el, "line", { x1: L, x2: W - R, y1: ys(v), y2: ys(v), stroke: "#2A3242" });
    svg(el, "text", { x: L - 8, y: ys(v) + 4, fill: "#98A2B3", "font-size": 12, "text-anchor": "end" }, v);
  }
  for (const deg of [0, 20, 40, 60, 80]) {
    svg(el, "text", { x: xs(deg), y: H - 16, fill: "#98A2B3", "font-size": 12, "text-anchor": "middle" }, `${deg}°`);
  }
  svg(el, "text", { x: L, y: 14, fill: "#98A2B3", "font-size": 12 }, "pixels per degree, by angle from the optical axis");
  const line = (key, max, color) => {
    const pts = d.deg.map((deg, i) => [deg, d[key][i]]).filter(([deg]) => deg <= max);
    return svg(el, "polyline", {
      class: "curve", fill: "none", stroke: color, "stroke-width": 3,
      points: pts.map(([deg, v]) => `${xs(deg).toFixed(1)},${ys(v).toFixed(1)}`).join(" "),
    });
  };
  const curves = { kb: line("kb", 85, "#F2F4F7"), p107: line("p107", d.half_107, "#FFB03B"), p364: line("p364", d.half_364, "#5AC8FA") };
  svg(el, "text", { x: xs(62), y: ys(d.kb[62]) - 10, fill: "#F2F4F7", "font-size": 13, "font-weight": 700 }, "native");
  svg(el, "text", { x: xs(8), y: ys(d.p107[8]) + 18, fill: "#FFB03B", "font-size": 13, "font-weight": 700 }, "keep field");
  svg(el, "text", { x: xs(20), y: ys(d.p364[20]) - 10, fill: "#5AC8FA", "font-size": 13, "font-weight": 700 }, "keep centre");
  const style = svg(el, "style");
  style.textContent = ".curve{opacity:.35;transition:opacity .2s,stroke-width .2s}.curve.on{opacity:1;stroke-width:5}";
  return { curves, bands };
}

/* Effect columns: the rolling-shutter divider and the two still-sequence scrubbers share one slider look. */
function initDividerRange(input) {
  const box = input.parentElement.querySelector("[data-compare]");
  input.addEventListener("input", () => box.style.setProperty("--x", `${(+input.value * 100).toFixed(2)}%`));
  new MutationObserver(() => {
    const x = parseFloat(box.style.getPropertyValue("--x"));
    if (!Number.isNaN(x)) input.value = x / 100;
  }).observe(box, { attributes: true, attributeFilter: ["style"] });
}

async function initEffectScrub(root) {
  const dir = root.dataset.scrub;
  const meta = await loadJSON(`${dir}meta.json`);
  const frames = preload([...Array(meta.n).keys()].map((i) => `${dir}${pad3(i)}.jpg`));
  const img = root.querySelector("img");
  const range = root.querySelector("input[type=range]");
  const val = root.querySelector(".ca-val");
  range.max = meta.n - 1;
  const show = (i) => {
    img.src = frames[i].src;
    range.value = i;
    if (val) val.textContent = `${(meta.strength[i] * 100).toFixed(1)} %`;
  };
  show(0);
  let i = 0, step = 1, running = false, stop = () => {};
  // Ping-pong only while on screen, until the viewer takes the slider.
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !running && !root.dataset.touched) {
      running = true;
      stop = autoplay(() => {
        if (i + step < 0 || i + step >= meta.n) step = -step;
        i += step;
        show(i);
      }, 60);
    } else if (!e.isIntersecting && running) { running = false; stop(); }
  }, { threshold: 0.2 }).observe(root);
  range.addEventListener("input", () => { root.dataset.touched = "1"; running = false; stop(); i = +range.value; show(i); });
}

/* Lens sweep per scene: plays as video; the slider follows it, and dragging it seeks to that field of view. */
async function initLens(root) {
  const scenes = await loadJSON("media/lens/scenes.json");
  const video = root.querySelector(".lens-view video");
  const tag = root.querySelector(".lens-fov");
  const play = root.querySelector(".lens-play");
  const range = root.querySelector("input[type=range]");
  const chips = root.querySelector(".chips");
  let scene = scenes[0], manual = false;

  const frame = () => Math.min(scene.p.length - 1, Math.floor(video.currentTime * 30));
  const tick = () => {
    const i = frame();
    tag.textContent = scene.label[i];
    if (!manual) range.value = Math.round(scene.p[i] * 1000);
    requestAnimationFrame(tick);
  };
  // Held at rate 0 while scrubbed: a paused <video> gets a native play overlay in some browsers.
  const resume = () => { manual = false; play.hidden = true; video.playbackRate = 1; video.play().catch(() => {}); };
  const select = (s, button) => {
    for (const b of chips.children) b.setAttribute("aria-selected", b === button ? "true" : "false");
    scene = s;
    pending = null;
    seeking = false;
    video.src = `media/lens/${s.id}.mp4`;
    resume();
  };
  scenes.forEach((s, k) => {
    const b = document.createElement("button");
    b.setAttribute("role", "tab");
    b.innerHTML = `<img src="media/lens/${s.id}.jpg" alt="Scene ${k + 1}" loading="lazy">`;
    b.addEventListener("click", () => select(s, b));
    chips.append(b);
  });
  // One seek in flight at a time; while it runs, only the latest slider position is kept.
  let pending = null, seeking = false;
  const seek = () => {
    if (pending === null || seeking) return;
    seeking = true;
    video.currentTime = pending;
    pending = null;
  };
  video.addEventListener("seeked", () => { seeking = false; seek(); });
  range.addEventListener("input", () => {
    manual = true;
    video.playbackRate = 0;
    play.hidden = false;
    // The frame of the forward half closest to the chosen lens.
    const target = +range.value / 1000;
    const peak = scene.p.indexOf(Math.max(...scene.p));
    let best = 0;
    for (let i = 0; i <= peak; i++) if (Math.abs(scene.p[i] - target) < Math.abs(scene.p[best] - target)) best = i;
    pending = (best + 0.5) / 30;
    seek();
  });
  play.addEventListener("click", resume);
  new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) video.pause();
    else video.play().catch(() => {});  // at rate 0 when scrubbed, so it stays on the chosen frame
  }, { threshold: 0.1 }).observe(video);
  select(scenes[0], chips.firstChild);
  requestAnimationFrame(tick);
}

/* Method figures. Each plays on its own until the viewer drags it; dragging never selects text. */
function interact(el, W, H, onMove, stop) {
  const at = (e) => {
    const r = el.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  };
  el.addEventListener("pointerdown", (e) => { e.preventDefault(); stop(); el.setPointerCapture(e.pointerId); onMove(at(e)); });
  el.addEventListener("pointermove", (e) => { if (el.hasPointerCapture(e.pointerId)) onMove(at(e)); });
}

/* 01: a Gaussian's sigma points through an equidistant fisheye, and the tiles they select. */
function initLensGlyph(el) {
  const W = 420, H = 230, cx = 210, cy = 115, f = 150, TILE = 16;
  const project = ([x, y]) => {  // pinhole-plane point to fisheye pixel
    const q = Math.hypot(x, y), k = q > 1e-9 ? (f * Math.atan(q)) / q : f;
    return [cx + k * x, cy + k * y];
  };
  const unproject = ([u, v]) => {
    const dx = u - cx, dy = v - cy, r = Math.hypot(dx, dy), k = r > 1e-9 ? Math.tan(Math.min(r / f, 1.32)) / r : 1 / f;
    return [k * dx, k * dy];
  };
  const pts = (list) => list.map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" ");
  const clip = svg(svg(el, "defs"), "clipPath", { id: "lens-clip" });
  svg(clip, "rect", { width: W, height: H, rx: 6 });
  const g = svg(el, "g", { "clip-path": "url(#lens-clip)" });
  svg(g, "rect", { width: W, height: H, fill: "#121722" });
  for (const c of [-4, -2, -1.2, -0.6, 0, 0.6, 1.2, 2, 4]) {
    const v = [], h = [];
    for (let s = -8; s <= 8.001; s += 0.08) { v.push(project([c, s])); h.push(project([s, c * 0.6])); }
    svg(g, "polyline", { points: pts(v), fill: "none", stroke: "#2A3242" });
    svg(g, "polyline", { points: pts(h), fill: "none", stroke: "#2A3242" });
  }
  const tiles = svg(g, "g");
  const outer = svg(g, "polygon", { fill: "rgba(255,176,59,.08)", stroke: "#FFB03B", "stroke-opacity": 0.35 });
  const inner = svg(g, "polygon", { fill: "rgba(255,176,59,.22)", stroke: "#FFB03B" });
  const marks = svg(g, "g");
  svg(el, "text", { x: 12, y: 20, fill: "#5AC8FA", "font-size": 13, "font-weight": 700 }, "tiles");
  svg(el, "text", { x: 52, y: 20, fill: "#FFB03B", "font-size": 13, "font-weight": 700 }, "sigma points");
  svg(el, "text", { x: W - 10, y: H - 10, fill: "#98A2B3", "font-size": 12, "text-anchor": "end" }, "fisheye image");
  const a = 0.5, b1 = [0.42 * Math.cos(a), 0.42 * Math.sin(a)], b2 = [-0.12 * Math.sin(a), 0.12 * Math.cos(a)];
  const set = (c) => {
    const at = (s, t) => project([c[0] + s * b1[0] + t * b2[0], c[1] + s * b1[1] + t * b2[1]]);
    const ring = (k) => Array.from({ length: 72 }, (_, i) => at(k * Math.cos(i * Math.PI / 36), k * Math.sin(i * Math.PI / 36)));
    inner.setAttribute("points", pts(ring(1)));
    outer.setAttribute("points", pts(ring(2)));
    const r3 = Math.sqrt(3), sp = [at(0, 0), at(r3, 0), at(-r3, 0), at(0, r3), at(0, -r3)];
    marks.replaceChildren(); tiles.replaceChildren();
    for (const p of sp.slice(1)) svg(marks, "line", { x1: sp[0][0], y1: sp[0][1], x2: p[0], y2: p[1], stroke: "#F2F4F7", "stroke-opacity": 0.5, "stroke-dasharray": "3 3" });
    for (const p of sp) svg(marks, "circle", { cx: p[0], cy: p[1], r: 4, fill: "#FFB03B", stroke: "#0E1116", "stroke-width": 1.5 });
    const xs = sp.map((p) => p[0]), ys = sp.map((p) => p[1]);
    const x0 = Math.floor(Math.min(...xs) / TILE) * TILE, x1 = Math.ceil(Math.max(...xs) / TILE) * TILE;
    const y0 = Math.floor(Math.min(...ys) / TILE) * TILE, y1 = Math.ceil(Math.max(...ys) / TILE) * TILE;
    for (let x = x0; x < x1; x += TILE) for (let y = y0; y < y1; y += TILE) {
      svg(tiles, "rect", { x: x + 0.5, y: y + 0.5, width: TILE - 1, height: TILE - 1, fill: "rgba(90,200,250,.13)", stroke: "rgba(90,200,250,.45)" });
    }
  };
  let p = 0;
  const stop = autoplay(() => { p += 0.025; set([Math.tan(0.62 + 0.6 * Math.sin(p)), 0.18]); }, 40);
  interact(el, W, H, (q) => set(unproject(q)), stop);
  set([Math.tan(0.62), 0.18]);
}

/* 02: a 4D Gaussian in the (t, x) plane, sliced at tau. */
function initSlice(el) {
  const W = 420, H = 240, T0 = 40, T1 = 404, X0 = 16, X1 = 206;
  const mt = 222, mx = 110, st = 60, sx = 46, rho = 0.86;  // plot pixels
  const c = rho * st * sx;
  const sd = Math.sqrt(sx * sx - (c * c) / (st * st));
  const a = st * st, d = sx * sx, tr = a + d, det = a * d - c * c;
  const l1 = tr / 2 + Math.sqrt((tr * tr) / 4 - det), l2 = tr / 2 - Math.sqrt((tr * tr) / 4 - det);
  const ang = (Math.atan2(l1 - a, c) * 180) / Math.PI;
  svg(el, "line", { x1: T0, x2: T1, y1: X1, y2: X1, stroke: "#2A3242" });
  svg(el, "line", { x1: T0, x2: T0, y1: X0, y2: X1, stroke: "#2A3242" });
  svg(el, "text", { x: T1, y: X1 + 18, fill: "#98A2B3", "font-size": 13, "text-anchor": "end" }, "time t");
  svg(el, "text", { x: T0 - 8, y: X0 + 10, fill: "#98A2B3", "font-size": 13, "text-anchor": "end" }, "x");
  for (const k of [2, 1]) {
    svg(el, "ellipse", {
      cx: mt, cy: mx, rx: k * Math.sqrt(l1), ry: k * Math.sqrt(l2), transform: `rotate(${ang} ${mt} ${mx})`,
      fill: k === 1 ? "rgba(255,176,59,.22)" : "rgba(255,176,59,.08)", stroke: "#FFB03B", "stroke-opacity": k === 1 ? 0.9 : 0.35,
    });
  }
  const v = c / (st * st);  // path of the conditional mean: constant velocity
  svg(el, "line", { x1: T0, x2: T1, y1: mx + v * (T0 - mt), y2: mx + v * (T1 - mt), stroke: "#98A2B3", "stroke-dasharray": "4 5" });
  const cut = svg(el, "line", { y1: X0, y2: X1, stroke: "#F2F4F7", "stroke-width": 1.5 });
  const bell = svg(el, "polyline", { fill: "rgba(90,200,250,.25)", stroke: "#5AC8FA", "stroke-width": 2.5 });
  const dot = svg(el, "circle", { r: 4.5, fill: "#5AC8FA" });
  const tl = svg(el, "text", { y: X1 + 18, fill: "#F2F4F7", "font-size": 14, "font-style": "italic", "text-anchor": "middle" }, "τ");
  const gate = svg(el, "text", { x: T1, y: 18, fill: "#5AC8FA", "font-size": 14, "text-anchor": "end", "font-weight": 700 });
  const set = (tau) => {
    const m = mx + v * (tau - mt);
    const w = Math.exp(-((tau - mt) ** 2) / (2 * st * st));
    const pts = [];
    for (let x = X0; x <= X1; x += 2) pts.push(`${(tau + 56 * w * Math.exp(-((x - m) ** 2) / (2 * sd * sd))).toFixed(1)},${x}`);
    bell.setAttribute("points", `${tau},${X0} ${pts.join(" ")} ${tau},${X1}`);
    cut.setAttribute("x1", tau);
    cut.setAttribute("x2", tau);
    dot.setAttribute("cx", tau); dot.setAttribute("cy", m);
    tl.setAttribute("x", tau);
    gate.textContent = `gate w_t = ${w.toFixed(2)}`;
  };
  let p = 0;
  const stop = autoplay(() => { p += 0.03; set(mt + Math.sin(p) * 150); }, 40);
  interact(el, W, H, ([x]) => set(clamp(x, T0 + 4, T1 - 4)), stop);
  set(mt);
}

/* 03 and 04: the thin axis of a conditioned Gaussian in emulated fp32, from the covariance and from its root.
   One spatial axis and time: long axis 1 at angle th in the (x, t) plane, thin axis r. */
const f32 = Math.fround;
function thinAxis(r, th) {
  const c = Math.cos(th), s = Math.sin(th);
  const x0 = f32(c), x1 = f32(-r * s), m0 = f32(s), m1 = f32(r * c);  // rows of M = RS: space, time
  const stt = f32(f32(m0 * m0) + f32(m1 * m1));
  const sxx = f32(f32(x0 * x0) + f32(x1 * x1)), sxt = f32(f32(x0 * m0) + f32(x1 * m1));
  const v = f32(sxx - f32(f32(sxt * sxt) / stt));
  const n = f32(Math.sqrt(stt)), h0 = f32(m0 / n), h1 = f32(m1 / n), dt = f32(f32(x0 * h0) + f32(x1 * h1));
  const n0 = f32(x0 - f32(dt * h0)), n1 = f32(x1 - f32(dt * h1));
  return {
    truth: Math.abs(x0 * m1 - x1 * m0) / Math.sqrt(m0 * m0 + m1 * m1),  // fp64 from the same stored parameters
    sigma: v > 0 ? f32(Math.sqrt(v)) : NaN,
    root: f32(Math.sqrt(f32(f32(n0 * n0) + f32(n1 * n1)))),
  };
}

/* A fan of sparks, each a fast, thin Gaussian drawn at the width a route returns relative to the true width.
   A spark whose variance comes out negative is not a covariance and is drawn as a dashed outline. */
function initSparks(fig) {
  const el = fig.querySelector("svg"), range = fig.querySelector("input");
  const routes = fig.dataset.sparks.split(" ");
  const W = 420, H = 230, n = routes.length, pw = W / n, BASE = 1.4;
  const COL = { sigma: "#FF6B6B", root: "#7EE081" }, NAME = { sigma: "from Σ₃D", root: "from N" };
  let seed = 11;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  // Each spark spans a fraction of the panel along its direction: [u0, u1] of the reach to the panel edge.
  const sparks = Array.from({ length: 34 }, () => {
    const u1 = 0.5 + rnd() * 0.48;
    return {
      ang: -(10 + rnd() * 66) * Math.PI / 180, u0: u1 * (0.08 + rnd() * 0.3), u1,
      th: Math.PI / 4 + (rnd() - 0.5) * 0.9, mul: 10 ** ((rnd() - 0.5) * 0.9),
    };
  });
  const defs = svg(el, "defs");
  const glow = svg(defs, "radialGradient", { id: `glow-${n}` });
  svg(glow, "stop", { offset: 0, "stop-color": "#FFF3D6", "stop-opacity": 1 });
  svg(glow, "stop", { offset: 1, "stop-color": "#FFB03B", "stop-opacity": 0 });
  const panels = routes.map((route, i) => {
    const g = svg(el, "g", { transform: `translate(${i * pw} 0)` });
    svg(g, "rect", { x: 2, y: 2, width: pw - 4, height: H - 4, rx: 6, fill: "#0B0E13" });
    const o = [22, H - 34];
    svg(g, "circle", { cx: o[0], cy: o[1], r: 26, fill: `url(#glow-${n})` });
    const body = svg(g, "g");
    svg(g, "text", { x: 12, y: 22, fill: COL[route], "font-size": 13, "font-weight": 700 }, NAME[route]);
    const note = svg(g, "text", { x: 12, y: H - 10, fill: "#98A2B3", "font-size": 11 });
    return { route, body, note, o };
  });
  const set = (v) => {
    const e = -2 - 4 * v;  // thin axis from 1e-2 to 1e-6
    for (const { route, body, note, o } of panels) {
      body.replaceChildren();
      let dropped = 0, off = 0;
      for (const sp of sparks) {
        const a = thinAxis(clamp(10 ** e * sp.mul, 1e-7, 1), sp.th), got = a[route];
        const reach = Math.min((pw - 14 - o[0]) / Math.cos(sp.ang), (o[1] - 34) / -Math.sin(sp.ang));
        const mid = ((sp.u0 + sp.u1) / 2) * reach, half = ((sp.u1 - sp.u0) / 2) * reach;
        const c = [o[0] + Math.cos(sp.ang) * mid, o[1] + Math.sin(sp.ang) * mid];
        const at = { cx: c[0].toFixed(1), cy: c[1].toFixed(1), rx: half.toFixed(1), transform: `rotate(${(sp.ang * 180 / Math.PI).toFixed(1)} ${c[0].toFixed(1)} ${c[1].toFixed(1)})` };
        if (Number.isNaN(got)) {
          dropped++;
          svg(body, "ellipse", { ...at, ry: BASE, fill: "none", stroke: "#FF6B6B", "stroke-opacity": 0.55, "stroke-dasharray": "3 3" });
          continue;
        }
        const ratio = got / a.truth;
        if (Math.abs(ratio - 1) > 0.1) off++;
        const ry = BASE * clamp(ratio, 0.05, 5);
        svg(body, "ellipse", { ...at, ry: (ry * 3).toFixed(2), fill: "#FFB03B", "fill-opacity": 0.1 });
        svg(body, "ellipse", { ...at, ry: ry.toFixed(2), fill: "#FFE2A8", "fill-opacity": 0.95 });
      }
      note.textContent = dropped || off ? `${dropped} not a covariance · ${off} off by over 10%` : `all ${sparks.length} within 10%`;
    }
  };
  let p = 0;
  const stop = autoplay(() => { p += 0.012; range.value = Math.round(500 - 500 * Math.cos(p)); set(range.value / 1000); }, 50);
  range.addEventListener("input", () => { stop(); set(range.value / 1000); });
  interact(el, W, H, ([x]) => { range.value = Math.round(clamp((x % pw) / pw) * 1000); set(range.value / 1000); }, stop);
  set(0);
}

/* 05: a pixel ray through a 2D Gaussian and its densest point, found by whitening with the triangular factor. */
function initRay(el) {
  const W = 420, H = 220, mu = [262, 100], sxx = 70 ** 2, syy = 26 ** 2, ang = -0.5;
  const c = Math.cos(ang), s = Math.sin(ang);
  const S = [c * c * sxx + s * s * syy, c * s * (sxx - syy), s * s * sxx + c * c * syy];  // [xx, xy, yy]
  const l11 = Math.sqrt(S[0]), l21 = S[1] / l11, l22 = Math.sqrt(S[2] - l21 * l21);
  const solve = ([x, y]) => { const a = x / l11; return [a, (y - l21 * a) / l22]; };  // L^-1 by forward substitution
  for (const k of [2, 1]) {
    svg(el, "ellipse", {
      cx: mu[0], cy: mu[1], rx: k * 70, ry: k * 26, transform: `rotate(${(ang * 180) / Math.PI} ${mu[0]} ${mu[1]})`,
      fill: k === 1 ? "rgba(255,176,59,.22)" : "rgba(255,176,59,.08)", stroke: "#FFB03B", "stroke-opacity": k === 1 ? 0.9 : 0.35,
    });
  }
  // Contour of the density through l*: the ray touches it there. l* is the closest point after whitening,
  // not the foot of the perpendicular from the mean.
  const contour = svg(el, "ellipse", {
    cx: mu[0], cy: mu[1], transform: `rotate(${(ang * 180) / Math.PI} ${mu[0]} ${mu[1]})`,
    fill: "none", stroke: "#5AC8FA", "stroke-opacity": 0.75, "stroke-dasharray": "5 4",
  });
  const o = [30, 180];
  svg(el, "polygon", { points: `${o[0] - 16},${o[1] - 9} ${o[0] - 16},${o[1] + 9} ${o[0]},${o[1]}`, fill: "#98A2B3" });
  svg(el, "text", { x: o[0] - 14, y: o[1] + 26, fill: "#98A2B3", "font-size": 12 }, "pixel ray");
  const ray = svg(el, "line", { x1: o[0], y1: o[1], stroke: "#5AC8FA", "stroke-width": 2 });
  const dl = svg(el, "line", { stroke: "#F2F4F7", "stroke-dasharray": "3 4" });
  svg(el, "circle", { cx: mu[0], cy: mu[1], r: 3.5, fill: "#FFB03B" });
  const peak = svg(el, "circle", { r: 6, fill: "#5AC8FA", stroke: "#0E1116", "stroke-width": 1.5 });
  const lab = svg(el, "text", { fill: "#5AC8FA", "font-size": 14, "font-style": "italic" }, "ℓ*");
  const read = svg(el, "text", { x: W - 10, y: H - 12, fill: "#F2F4F7", "font-size": 13, "text-anchor": "end" });
  const set = (t) => {  // t: ray angle
    const d = [Math.cos(t), Math.sin(t)], p = [o[0] - mu[0], o[1] - mu[1]], ld = solve(d), lp = solve(p);
    const l = Math.max(0, -(ld[0] * lp[0] + ld[1] * lp[1]) / (ld[0] ** 2 + ld[1] ** 2));
    const q = [o[0] + l * d[0], o[1] + l * d[1]], r = solve([q[0] - mu[0], q[1] - mu[1]]);
    const D = Math.hypot(...r), alpha = Math.exp(-(D * D) / 2);
    contour.setAttribute("rx", 70 * D); contour.setAttribute("ry", 26 * D);
    ray.setAttribute("x2", o[0] + 460 * d[0]); ray.setAttribute("y2", o[1] + 460 * d[1]);
    dl.setAttribute("x1", mu[0]); dl.setAttribute("y1", mu[1]); dl.setAttribute("x2", q[0]); dl.setAttribute("y2", q[1]);
    peak.setAttribute("cx", q[0]); peak.setAttribute("cy", q[1]); peak.setAttribute("fill-opacity", (0.25 + 0.75 * alpha).toFixed(2));
    lab.setAttribute("x", q[0] + 8); lab.setAttribute("y", q[1] + 20);
    read.textContent = `D = ${D.toFixed(2)}   density at ℓ* = ${alpha.toFixed(2)}`;
  };
  let p = 0;
  const stop = autoplay(() => { p += 0.02; set(-0.3 + 0.32 * Math.sin(p)); }, 40);
  interact(el, W, H, ([x, y]) => set(Math.atan2(y - o[1], Math.max(x - o[0], 1))), stop);
  set(-0.3);
}

document.addEventListener("DOMContentLoaded", () => {
  renderMathInElement(document.body, {
    delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
  });
  // Keep punctuation that follows inline math on the same line.
  for (const k of document.querySelectorAll(".katex")) {
    const next = k.nextSibling;
    if (k.closest(".katex-display") || !next || next.nodeType !== 3 || !/^[,.;:)]/.test(next.data)) continue;
    const wrap = document.createElement("span");
    wrap.className = "nobr";
    k.replaceWith(wrap);
    wrap.append(k, next.data[0]);
    next.data = next.data.slice(1);
  }
});
initHero();
document.querySelectorAll("[data-compare]").forEach(initCompare);
document.querySelectorAll("video.media:not([data-manual])").forEach((v) => visible.observe(v));
visible.observe(document.querySelector("#closing video"));
initTradeoff(document.querySelector(".tradeoff"));
initLens(document.querySelector(".lens"));
document.querySelectorAll("[data-divider]").forEach(initDividerRange);
document.querySelectorAll(".effect[data-scrub]").forEach(initEffectScrub);
initLensGlyph(document.querySelector('[data-glyph="lens"]'));
initSlice(document.querySelector('[data-glyph="slice"]'));
document.querySelectorAll("[data-sparks]").forEach(initSparks);
initRay(document.querySelector('[data-glyph="ray"]'));

for (const b of document.querySelectorAll("[data-copy]")) {
  b.addEventListener("click", async () => {
    await navigator.clipboard.writeText(document.querySelector(b.dataset.copy).textContent);
    b.textContent = "Copied";
    setTimeout(() => { b.textContent = "Copy"; }, 1500);
  });
}
