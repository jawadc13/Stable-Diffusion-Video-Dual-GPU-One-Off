// Minimal ComfyUI-style node editor (vanilla JS, no build step).
const TYPE_COLORS = { IMAGE: "#64b5f6", FRAMES: "#81c784", VIDEO: "#ffb74d", STRING: "#ce93d8" };
const $ = (s) => document.querySelector(s);
const canvas = $("#canvas"), graphEl = $("#graph"), wiresEl = $("#wires");

let INFO = {};
let state = { nodes: {}, links: [], nextId: 1 };   // links: {from:[id,outIdx], to:[id,inputName]}
let view = { x: 40, y: 40, z: 1 };
let selected = null;
const els = {};  // id -> {root, inDots:{}, outDots:[], widgets:{}, prog, ptext, preview}

// ------------------------------------------------------------------ graph --
function addNode(cls, x, y, widgets = {}, id = null) {
  id = String(id ?? state.nextId++);
  state.nextId = Math.max(state.nextId, +id + 1);
  const def = INFO[cls];
  const w = {};
  for (const [k, spec] of Object.entries(def.widgets)) w[k] = widgets[k] ?? spec.default;
  state.nodes[id] = { id, cls, x, y, widgets: w };
  renderNode(state.nodes[id]);
  return id;
}

function removeNode(id) {
  state.links = state.links.filter((l) => l.from[0] !== id && l.to[0] !== id);
  els[id].root.remove(); delete els[id]; delete state.nodes[id];
  drawWires(); persist();
}

function connect(from, to) {
  state.links = state.links.filter((l) => !(l.to[0] === to[0] && l.to[1] === to[1]));
  state.links.push({ from, to });
  drawWires(); persist();
}

// ----------------------------------------------------------------- render --
function renderNode(n) {
  const def = INFO[n.cls];
  const root = document.createElement("div");
  root.className = "node";
  root.style.left = n.x + "px"; root.style.top = n.y + "px";
  root.innerHTML = `<div class="head"><span>${n.cls}</span><span class="cat">${def.category}</span></div><div class="body"></div>`;
  root.title = def.description;
  const body = root.querySelector(".body");
  const e = { root, inDots: {}, outDots: [], widgets: {} };

  for (const [name, inp] of Object.entries(def.inputs)) {
    const row = document.createElement("div");
    row.className = "sock in";
    row.innerHTML = `<span class="dot" style="background:${TYPE_COLORS[inp.type]}"></span><span class="${inp.optional ? "opt" : ""}">${name} <small>${inp.type}</small></span>`;
    const dot = row.querySelector(".dot");
    dot.dataset.type = inp.type;
    dot.addEventListener("mouseup", (ev) => { if (drag?.kind === "wire") { finishWire(n.id, name, inp.type); ev.stopPropagation(); } });
    dot.addEventListener("mousedown", (ev) => {  // click on input = disconnect
      ev.stopPropagation();
      const before = state.links.length;
      state.links = state.links.filter((l) => !(l.to[0] === n.id && l.to[1] === name));
      if (before !== state.links.length) { drawWires(); persist(); }
    });
    e.inDots[name] = dot; body.appendChild(row);
  }
  def.outputs.forEach(([name, type], i) => {
    const row = document.createElement("div");
    row.className = "sock out";
    row.innerHTML = `<span><small>${type}</small> ${name}</span><span class="dot" style="background:${TYPE_COLORS[type]}"></span>`;
    const dot = row.querySelector(".dot");
    dot.addEventListener("mousedown", (ev) => { ev.stopPropagation(); drag = { kind: "wire", from: [n.id, i], type }; });
    e.outDots.push(dot); body.appendChild(row);
  });

  for (const [k, spec] of Object.entries(def.widgets)) {
    const row = document.createElement("div");
    row.className = "w" + (spec.type === "text" || spec.type === "image" ? " col" : "");
    row.innerHTML = `<label>${k}</label>`;
    let input;
    if (spec.type === "combo") {
      input = document.createElement("select");
      for (const o of spec.options) input.add(new Option(o, o));
    } else if (spec.type === "text") {
      input = document.createElement("textarea");
    } else if (spec.type === "image") {
      input = document.createElement("div");
      input.innerHTML = `<input type="file" accept="image/*"><img>`;
      const img = input.querySelector("img");
      const show = () => { img.src = n.widgets[k] ? `/view/input/${encodeURIComponent(n.widgets[k])}` : ""; img.style.display = n.widgets[k] ? "" : "none"; };
      show();
      input.querySelector("input").addEventListener("change", async (ev) => {
        const fd = new FormData(); fd.append("image", ev.target.files[0]);
        const r = await (await fetch("/upload", { method: "POST", body: fd })).json();
        n.widgets[k] = r.name; show(); persist();
      });
      row.appendChild(input); body.appendChild(row); continue;
    } else {
      input = document.createElement("input");
      if (spec.type === "int" || spec.type === "float") {
        input.type = "number";
        if (spec.min !== undefined) input.min = spec.min;
        if (spec.max !== undefined) input.max = spec.max;
        input.step = spec.step ?? (spec.type === "int" ? 1 : 0.01);
      } else if (spec.password) input.type = "password";
    }
    input.value = n.widgets[k];
    input.addEventListener("change", () => {
      let v = input.value;
      if (spec.type === "int") v = parseInt(v); else if (spec.type === "float") v = parseFloat(v);
      n.widgets[k] = v; persist();
    });
    input.addEventListener("mousedown", (ev) => ev.stopPropagation());
    input.addEventListener("wheel", (ev) => ev.stopPropagation());
    row.appendChild(input); body.appendChild(row); e.widgets[k] = input;
  }

  const prog = document.createElement("div"); prog.className = "prog"; prog.innerHTML = "<div></div>";
  const ptext = document.createElement("div"); ptext.className = "ptext";
  const preview = document.createElement("div");
  body.append(prog, ptext, preview);
  Object.assign(e, { prog: prog.firstChild, ptext, preview });

  root.querySelector(".head").addEventListener("mousedown", (ev) => {
    ev.stopPropagation(); select(n.id);
    drag = { kind: "node", id: n.id, sx: ev.clientX, sy: ev.clientY, nx: n.x, ny: n.y };
  });
  root.addEventListener("mousedown", () => select(n.id));
  graphEl.appendChild(root); els[n.id] = e;
}

function select(id) {
  if (selected && els[selected]) els[selected].root.classList.remove("selected");
  selected = id;
  if (id) els[id].root.classList.add("selected");
}

function dotPos(dot) {
  const r = dot.getBoundingClientRect(), c = canvas.getBoundingClientRect();
  return [(r.left + r.width / 2 - c.left - view.x) / view.z, (r.top + r.height / 2 - c.top - view.y) / view.z];
}

function curve([x1, y1], [x2, y2]) {
  const d = Math.max(40, Math.abs(x2 - x1) / 2);
  return `M${x1},${y1} C${x1 + d},${y1} ${x2 - d},${y2} ${x2},${y2}`;
}

let tempWire = null;
function drawWires() {
  let html = "";
  for (const l of state.links) {
    const a = els[l.from[0]]?.outDots[l.from[1]], b = els[l.to[0]]?.inDots[l.to[1]];
    if (!a || !b) continue;
    const type = INFO[state.nodes[l.from[0]].cls].outputs[l.from[1]][1];
    html += `<path d="${curve(dotPos(a), dotPos(b))}" stroke="${TYPE_COLORS[type]}"/>`;
  }
  if (tempWire) html += `<path d="${curve(tempWire.a, tempWire.b)}" stroke="#fff" stroke-dasharray="6 4"/>`;
  wiresEl.innerHTML = html;
}

function applyView() {
  const t = `translate(${view.x}px,${view.y}px) scale(${view.z})`;
  graphEl.style.transform = t; wiresEl.style.transform = t;
  canvas.style.backgroundPosition = `${view.x}px ${view.y}px`;
  canvas.style.backgroundSize = `${20 * view.z}px ${20 * view.z}px`;
  drawWires();
}

// ------------------------------------------------------------ interaction --
let drag = null;
function finishWire(toId, inputName, inType) {
  if (drag.type === inType && drag.from[0] !== toId) connect(drag.from, [toId, inputName]);
  drag = null; tempWire = null; drawWires();
}

canvas.addEventListener("mousedown", (ev) => {
  if (ev.button !== 0) return;
  hideMenu(); select(null);
  drag = { kind: "pan", sx: ev.clientX, sy: ev.clientY, vx: view.x, vy: view.y };
});
window.addEventListener("mousemove", (ev) => {
  if (!drag) return;
  if (drag.kind === "pan") { view.x = drag.vx + ev.clientX - drag.sx; view.y = drag.vy + ev.clientY - drag.sy; applyView(); }
  else if (drag.kind === "node") {
    const n = state.nodes[drag.id];
    n.x = drag.nx + (ev.clientX - drag.sx) / view.z; n.y = drag.ny + (ev.clientY - drag.sy) / view.z;
    els[n.id].root.style.left = n.x + "px"; els[n.id].root.style.top = n.y + "px"; drawWires();
  } else if (drag.kind === "wire") {
    const c = canvas.getBoundingClientRect();
    tempWire = { a: dotPos(els[drag.from[0]].outDots[drag.from[1]]), b: [(ev.clientX - c.left - view.x) / view.z, (ev.clientY - c.top - view.y) / view.z] };
    drawWires();
  }
});
window.addEventListener("mouseup", () => {
  if (drag?.kind === "node") persist();
  if (drag?.kind === "wire") { tempWire = null; drawWires(); }
  drag = null;
});
canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const c = canvas.getBoundingClientRect(), mx = ev.clientX - c.left, my = ev.clientY - c.top;
  const z = Math.min(2, Math.max(0.25, view.z * (ev.deltaY < 0 ? 1.1 : 0.9)));
  view.x = mx - (mx - view.x) * (z / view.z); view.y = my - (my - view.y) * (z / view.z); view.z = z;
  applyView();
}, { passive: false });
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Delete" && selected && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) { removeNode(selected); selected = null; }
  if (ev.key === "Enter" && ev.ctrlKey) queuePrompt();
});

// --------------------------------------------------------------- add menu --
const menu = $("#menu"), menuSearch = $("#menu-search"), menuList = $("#menu-list");
let menuAt = [0, 0];
function showMenu(ev) {
  ev.preventDefault();
  const c = canvas.getBoundingClientRect();
  menuAt = [(ev.clientX - c.left - view.x) / view.z, (ev.clientY - c.top - view.y) / view.z];
  menu.style.left = ev.clientX + "px"; menu.style.top = ev.clientY + "px";
  menu.classList.remove("hidden"); menuSearch.value = ""; fillMenu(); menuSearch.focus();
}
function hideMenu() { menu.classList.add("hidden"); }
function fillMenu() {
  const q = menuSearch.value.toLowerCase();
  menuList.innerHTML = "";
  for (const [name, d] of Object.entries(INFO).sort((a, b) => a[1].category.localeCompare(b[1].category))) {
    if (q && !name.toLowerCase().includes(q)) continue;
    const it = document.createElement("div");
    it.className = "item"; it.innerHTML = `${name}<small>${d.category}</small>`;
    it.onclick = () => { addNode(name, ...menuAt); hideMenu(); persist(); };
    menuList.appendChild(it);
  }
}
menuSearch.addEventListener("input", fillMenu);
menuSearch.addEventListener("keydown", (ev) => { if (ev.key === "Enter") menuList.firstChild?.click(); if (ev.key === "Escape") hideMenu(); });
menu.addEventListener("mousedown", (ev) => ev.stopPropagation());
canvas.addEventListener("contextmenu", showMenu);
canvas.addEventListener("dblclick", (ev) => { if (ev.target === canvas) showMenu(ev); });

// -------------------------------------------------------------- workflows --
function serialize() { return { nodes: Object.values(state.nodes), links: state.links, view }; }
function loadWorkflow(wf) {
  for (const id of Object.keys(state.nodes)) { els[id].root.remove(); delete els[id]; }
  state = { nodes: {}, links: [], nextId: 1 };
  for (const n of wf.nodes) if (INFO[n.cls]) addNode(n.cls, n.x, n.y, n.widgets, n.id);
  state.links = wf.links.filter((l) => state.nodes[l.from[0]] && state.nodes[l.to[0]]);
  if (wf.view) view = wf.view;
  requestAnimationFrame(applyView); persist();
}
function persist() { try { localStorage.setItem("dualgpu.workflow", JSON.stringify(serialize())); } catch {} }

function toPrompt() {
  const g = {};
  for (const n of Object.values(state.nodes)) g[n.id] = { class: n.cls, widgets: n.widgets, inputs: {} };
  for (const l of state.links) g[l.to[0]].inputs[l.to[1]] = l.from;
  return g;
}
async function queuePrompt() {
  for (const e of Object.values(els)) { e.root.classList.remove("error", "running"); e.prog.style.width = "0"; e.ptext.textContent = ""; }
  const r = await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ graph: toPrompt() }) });
  setStatus("queued " + (await r.json()).prompt_id.slice(0, 8));
}

$("#queue").onclick = queuePrompt;
$("#save").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(serialize(), null, 2)], { type: "application/json" }));
  a.download = "workflow.json"; a.click();
};
$("#load").onclick = () => $("#loadfile").click();
$("#loadfile").onchange = async (ev) => loadWorkflow(JSON.parse(await ev.target.files[0].text()));
$("#default").onclick = async () => loadWorkflow(await (await fetch("/web/default_workflow.json")).json());
$("#clear").onclick = () => loadWorkflow({ nodes: [], links: [] });

// -------------------------------------------------------------- websocket --
let running = null;
function setStatus(msg, err) { const s = $("#status"); s.textContent = msg; s.className = err ? "err" : ""; }
function log(msg) { const l = $("#log"); l.textContent = (new Date().toLocaleTimeString() + " " + msg + "\n" + l.textContent).slice(0, 8000); }
const gpuState = {};
function renderGpus() {
  $("#gpus").innerHTML = Object.entries(gpuState).map(([k, g]) =>
    `<div class="gpu"><b>${k}</b><small>${g.name ?? ""} · ${g.vram_gb ?? "?"} GB</small>
     <div class="prog" style="margin:4px 0"><div style="width:${g.pct ?? 0}%"></div></div><small>${g.text ?? "idle"}</small></div>`).join("");
}
function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onclose = () => setTimeout(connectWS, 1500);
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    switch (d.type) {
      case "gpus": for (const [k, v] of Object.entries(d.gpus)) gpuState[k] = { ...gpuState[k], ...v }; renderGpus(); break;
      case "gpu_progress":
        gpuState[d.gpu] = { ...gpuState[d.gpu], pct: (100 * d.step) / d.steps, text: `${d.step}/${d.steps}` };
        renderGpus(); break;
      case "executing":
        if (running && els[running]) els[running].root.classList.remove("running");
        running = d.node; if (running && els[running]) els[running].root.classList.add("running"); break;
      case "progress": {
        const e = els[d.node]; if (!e) break;
        if (d.steps) e.prog.style.width = (100 * d.step) / d.steps + "%";
        if (d.text) e.ptext.textContent = d.text; break;
      }
      case "log": log(`#${d.node} ${d.msg}`); break;
      case "executed": {
        const e = els[d.node];
        if (e && d.output.video) e.preview.innerHTML = `<video src="/view/output/${encodeURIComponent(d.output.video)}" controls loop autoplay muted></video>`;
        break;
      }
      case "status": setStatus(d.msg); log(d.msg); for (const g of Object.values(gpuState)) { g.pct = 0; g.text = "idle"; } renderGpus(); break;
      case "error": setStatus("error", true); log("ERROR: " + d.msg); if (running && els[running]) els[running].root.classList.add("error"); break;
    }
  };
}

// ------------------------------------------------------------------- boot --
(async () => {
  INFO = await (await fetch("/object_info")).json();
  const sys = await (await fetch("/system_info")).json();
  for (const n of sys.names) gpuState[n] = { ...sys.gpus[n] };
  renderGpus();
  let wf = null;
  try { wf = JSON.parse(localStorage.getItem("dualgpu.workflow")); } catch {}
  if (!wf || !wf.nodes?.length) wf = await (await fetch("/web/default_workflow.json")).json();
  loadWorkflow(wf);
  connectWS();
})();
