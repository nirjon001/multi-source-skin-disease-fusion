"""
CSE475 Skin Disease Classification - Grad-CAM visual inspection gallery (Phase 1)

Runs a trained checkpoint over images (ImageFolder layout) and emits a
self-contained browsable HTML gallery:
  - original thumbnail, Grad-CAM heatmap overlay per image
  - ground-truth label, model prediction + confidence
  - correctness badge (model agrees / disagrees with the folder label)
  - per-image marking (ok / wrong-label / uncertain + note) saved to
    browser localStorage, export/import-able as JSON, so you can flag
    suspicious images for a label-noise pass.

Usage:
  py -3.11 src/gradcam_gallery.py --checkpoint results/phase1_baseline_best.pt \
      --data F:/Downloads/skin_disease_images --splits test \
      --out results/gallery_phase1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_grad_enabled(True)
import timm
from PIL import Image
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def eval_tf(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def resolve_device(choice: str):
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device("cpu")


def model_classes_from_ckpt(ckpt) -> list[str]:
    """Rebuild ImageFolder class order (alphabetical) from the training data_root."""
    cfg = ckpt.get("config") or {}
    d = cfg.get("data_root")
    if not d:
        raise SystemExit("[FATAL] checkpoint has no config.data_root; pass --class-names a,b,c")
    root = Path(d)
    train_dir = next((root / s for s in ("train", "Train") if (root / s).exists()), root)
    ds = datasets.ImageFolder(str(train_dir))
    return ds.classes


def jet(t: np.ndarray) -> np.ndarray:
    """Manual jet colormap (no matplotlib dependency). t in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_batch(model, x_batch, device):
    """Grad-CAM over a batch. Returns CAM [B,H,W] in original 224 space, predicted
    indices [B], and logits [B,C] (one backward pass for the whole chunk)."""
    from torch.nn.functional import relu, interpolate, softmax
    acts = {}
    grads = {}

    def fwd_hook(mod, inp, out):
        acts["val"] = out

    def bwd_hook(mod, gin, gout):
        grads["val"] = gout[0]

    h1 = model.conv_head.register_forward_hook(fwd_hook)
    h2 = model.conv_head.register_full_backward_hook(bwd_hook)

    logits = model(x_batch)
    pred = logits.argmax(dim=1)
    scores = logits.gather(1, pred.unsqueeze(1)).squeeze(1)
    loss = scores.sum()
    model.zero_grad(set_to_none=True)
    loss.backward()

    h1.remove()
    h2.remove()

    a = acts["val"].detach()      # [B,C,H,W]
    g = grads["val"].detach()     # [B,C,H,W]
    weights = g.mean(dim=(2, 3), keepdim=True)          # [B,C,1,1]
    cam = relu((a * weights).sum(dim=1, keepdim=True))   # [B,1,H,W]
    cam = interpolate(cam, size=(x_batch.shape[2], x_batch.shape[3]),
                      mode="bilinear", align_corners=False)  # [B,1,224,224]
    cam = cam[:, 0]                                       # [B,H,W]
    cam = cam - cam.min(dim=2, keepdim=True).values.min(dim=1, keepdim=True).values
    amax = cam.max(dim=2, keepdim=True).values.max(dim=1, keepdim=True).values + 1e-8
    cam = cam / amax
    return cam.detach().cpu().numpy(), pred.cpu().tolist(), logits.detach()


# ---------------------------------------------------------------------------
# Gallery building
# ---------------------------------------------------------------------------
def build_thumb_gallery(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "cams").mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_rel(p: Path, split: str) -> str:
    return str(p).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM inspection gallery")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None,
                    help="ImageFolder root (train+val+test) or a flat class root")
    ap.add_argument("--splits", default="auto",
                    help="comma list: train,validation,test (default auto-detect)")
    ap.add_argument("--out", default="results/gallery_phase1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-width", type=int, default=320,
                    help="thumbnail + overlay long edge in px")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split (0 = all)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = model_classes_from_ckpt(ckpt)
    cfg = ckpt.get("config") or {}
    model_name = cfg.get("model", "efficientnet_b0")
    img_size = cfg.get("img_size", 224)

    device = resolve_device(args.device)
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[GALLERY] model={model_name} classes={classes} device={device}")

    data_root = Path(args.data if args.data else cfg.get("data_root", "."))
    if not data_root.exists():
        raise SystemExit(f"[FATAL] --data root not found: {data_root}")

    known = {"train", "Train", "validation", "val", "Val", "test", "Test"}
    present = {d.name: d for d in data_root.iterdir() if d.is_dir() and d.name in known}
    # If data root has no split folders but has class dirs, treat itself as one demo split.
    if not present:
        subdirs = [d for d in data_root.iterdir() if d.is_dir()]
        if subdirs:
            present = {".": data_root}

    if args.splits == "auto":
        wanted = list(present.keys())
    else:
        wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    wanted = [s for s in wanted if s in present]
    if not wanted:
        raise SystemExit(f"[FATAL] no usable split dirs under {data_root} (found {sorted(present)})")

    tf = eval_tf(img_size)
    out_dir = Path(args.out)
    build_thumb_gallery(out_dir)

    all_rows = []
    split_stats = {}
    for split in wanted:
        split_dir = present[split]
        ds = datasets.ImageFolder(str(split_dir), tf)
        n = len(ds)
        print(f"[GALLERY] split '{split}': {n} images")
        rows = []

        for i in range(n):
            if args.limit and i >= args.limit:
                break
            img_t, y = ds[i]
            path, cls_idx = ds.samples[i]
            path = Path(path)
            x = img_t.unsqueeze(0).to(device)
            x.requires_grad_(True)
            cam, pred, logits = gradcam_batch(model, x, device)
            cam0 = cam[0]
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            conf = float(probs[pred[0]])
            gt = classes[y]

            # load original for thumbnail
            with Image.open(path) as im0:
                im = im0.convert("RGB")
                w, h = im.size
                scale = args.max_width / max(w, h)
                thumb = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                thumb_img = np.asarray(thumb).astype(np.float32) / 255.0

            thumb_h, thumb_w = thumb_img.shape[:2]
            cam_res = Image.fromarray((cam0 * 255).astype(np.uint8)).resize((thumb_w, thumb_h), Image.BILINEAR)
            alpha0 = np.asarray(cam_res).astype(np.float32) / 255.0          # [H,W]
            # percentile-clip so only the salient region lights up (EfficientNet
            # last-layer Grad-CAM is diffuse; p40..p97 emphasises the hotspot)
            lo, hi = np.percentile(alpha0, 40), np.percentile(alpha0, 97)
            alpha = np.clip((alpha0 - lo) / max(hi - lo, 1e-6), 0, 1)
            heat = jet(alpha)                                               # [H,W,3]
            blend = thumb_img * (1.0 - 0.65 * alpha[..., None]) + heat * (0.65 * alpha[..., None])
            blend = np.clip(blend * 255, 0, 255).astype(np.uint8)

            rel = f"{split}/{path.relative_to(split_dir)}".replace("\\", "/")
            thumb_rel = f"thumbs/{rel}"
            cam_rel = f"cams/{rel}"
            (out_dir / thumb_rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / cam_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((thumb_img * 255).astype(np.uint8)).save(out_dir / thumb_rel, quality=88)
            Image.fromarray(blend).save(out_dir / cam_rel, quality=88)

            rows.append({
                "id": rel,
                "path": str(path),
                "split": split,
                "cls": gt,
                "pred": classes[pred[0]],
                "conf": round(conf, 4),
                "w": w, "h": h,
                "thumb": thumb_rel, "cam": cam_rel,
                "probs": {classes[c]: round(float(probs[c]), 4) for c in range(len(classes))},
            })
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{n}")

        correct = sum(1 for r in rows if r["cls"] == r["pred"])
        split_stats[split] = {"n": len(rows),
                              "acc": round(correct / len(rows), 4) if rows else 0.0,
                              "correct": correct}
        all_rows.extend(rows)

    write_html(out_dir, all_rows, split_stats, args.checkpoint, classes, device)

    print()
    for k, v in split_stats.items():
        print(f"[GALLERY] {k}: acc={v['acc']} ({v['correct']}/{v['n']})")
    print(f"[GALLERY] -> open {out_dir / 'gallery.html'} in your browser")


def write_html(out_dir: Path, rows, split_stats, checkpoint, classes, device) -> None:
    data = json.dumps({
        "checkpoint": str(checkpoint),
        "classes": classes,
        "device": str(device),
        "split_stats": split_stats,
        "rows": rows,
    })
    html = HTML_TEMPLATE.replace("/*__DATA__*/", data)
    (out_dir / "gallery.html").write_text(html, encoding="utf-8")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CSE475 Grad-CAM Gallery</title>
<style>
  :root{--bg:#0f1115;--card:#1a1e27;--line:#2a3040;--txt:#e6e9f0;--dim:#8a93a6;
        --ok:#2ecc71;--bad:#ff5c5c;--maybe:#f1c40f;--acc:#4f9fff}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);
    font-family:system-ui,Segoe UI,Roboto,sans-serif}
  header{position:sticky;top:0;background:#13161d;border-bottom:1px solid var(--line);
    padding:10px 16px;z-index:10}
  h1{font-size:15px;margin:0 0 8px} h1 small{color:var(--dim);font-weight:400}
  #stats{font-size:12px;color:var(--dim);margin-bottom:8px}
  #stats b{color:var(--txt)}
  .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:12px}
  .bar select,.bar input,#search{background:#20242e;border:1px solid var(--line);
    color:var(--txt);border-radius:6px;padding:5px 8px;font-size:12px}
  .bar button{background:#20242e;border:1px solid var(--line);color:var(--txt);
    border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px}
  .bar button:hover,#exp:hover{background:#2a3040}
  .bar button.on{background:var(--acc);border-color:var(--acc);color:#fff}
  #exp{background:#17a2b8;border:none;color:#fff}
  #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
    gap:12px;padding:16px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;
    overflow:hidden;display:flex;flex-direction:column}
  .imgs{position:relative;display:flex}
  .imgs img{width:50%;height:auto;display:block;min-height:120px;background:#000}
  .badge{position:absolute;top:6px;left:6px;padding:2px 7px;border-radius:20px;
    font-size:11px;font-weight:700}
  .b-ok{background:var(--ok)} .b-bad{background:var(--bad)} .b-na{background:#555}
  .body{padding:8px 10px 10px;font-size:12px;display:flex;flex-direction:column;gap:6px;flex:1}
  .fn{font-size:11px;color:var(--dim);word-break:break-all;min-height:14px;max-height:26px;overflow:hidden}
  .row{display:flex;justify-content:space-between;gap:6px}
  .lab{color:var(--dim)} .gt{color:var(--txt);font-weight:600}
  .pred{font-weight:700}
  .p-ok{color:var(--ok)} .p-bad{color:var(--bad)}
  .conf{color:var(--dim);font-size:11px}
  .marks{display:flex;gap:6px;margin-top:2px}
  .marks button{flex:1;border:1px solid var(--line);background:#20242e;color:var(--dim);
    border-radius:6px;padding:4px 0;cursor:pointer;font-size:12px}
  .marks button.m-ok.on{background:var(--ok);border-color:var(--ok);color:#000}
  .marks button.m-bad.on{background:var(--bad);border-color:var(--bad);color:#fff}
  .marks button.m-maybe.on{background:var(--maybe);border-color:var(--maybe);color:#000}
  .note{background:#20242e;border:1px solid var(--line);color:var(--txt);
    border-radius:6px;padding:4px 7px;font-size:11px;width:100%}
  #empty{padding:40px;text-align:center;color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>CSE475 Grad-CAM Gallery <small id="meta"></small></h1>
  <div id="stats"></div>
  <div class="bar">
    <select id="f-split"><option value="">split: all</option></select>
    <select id="f-cls"><option value="">gt: all</option></select>
    <select id="f-pred"><option value="">pred: all</option></select>
    <select id="f-mark"><option value="">marked: any</option>
      <option value="none">unmarked</option><option value="ok">ok</option>
      <option value="bad">wrong-label</option><option value="maybe">uncertain</option></select>
    <label><input type="checkbox" id="f-wrong"> only wrong</label>
    <label>conf ≥ <input type="number" id="f-conf" value="0" min="0" max="1" step="0.05" style="width:52px"></label>
    <input id="search" placeholder="search file name..." style="min-width:160px"/>
    <button id="exp">Export marks</button>
    <button id="imp-btn">Import marks</button>
    <input id="imp" type="file" accept=".json" style="display:none"/>
    <button id="clear-marks">Clear marks</button>
    <button id="toggle-all">Show/hide CAMs</button>
  </div>
</header>
<div id="grid"></div>
<div id="empty" style="display:none">No images match filters.</div>

<script>
const DATA = /*__DATA__*/;
const LS = "cse475_gallery_marks_v1";
let marks = {}; try{marks = JSON.parse(localStorage.getItem(LS) || "{}")}catch(e){}
document.getElementById("meta").textContent =
  " — " + DATA.checkpoint.split(/[\\\\/]/).pop() + " · classes: " + DATA.classes.join(", ");

function save(){localStorage.setItem(LS, JSON.stringify(marks))}
let showCams = true;

function statsHTML(){
  let h = "";
  for(const [k,v] of Object.entries(DATA.split_stats)){
    h += `<b>${k}</b>: ${v.acc} (${v.correct}/${v.n})&nbsp;&nbsp;`;
  }
  const total = DATA.rows.length, corr = DATA.rows.filter(r=>r.cls===r.pred).length;
  h += `<b>total</b>: ${total} <b>wrong</b>: ${total-corr}`;
  const m = Object.keys(marks).length;
  if(m) h += ` <b>marked</b>: ${m}`;
  return h;
}
document.getElementById("stats").innerHTML = statsHTML();

for(const v of [...new Set(DATA.rows.map(r=>r.split))])
  document.getElementById("f-split").insertAdjacentHTML("beforeend", `<option>${v}</option>`);
for(const v of DATA.classes){
  document.getElementById("f-cls").insertAdjacentHTML("beforeend", `<option>${v}</option>`);
  document.getElementById("f-pred").insertAdjacentHTML("beforeend", `<option>${v}</option>`);
}

function cardClone(rec){
  const card = document.createElement("div"); card.className = "card";
  card.dataset.id = rec.id;
  const ok = rec.cls === rec.pred;
  const mk = marks[rec.id] || {};
  const mark = mk.mark || "";

  const badge = document.createElement("div");
  badge.className = "badge " + (ok ? "b-ok" : "b-bad");
  badge.textContent = ok ? "MATCH" : "WRONG";

  const imgs = document.createElement("div"); imgs.className = "imgs";
  imgs.appendChild(badge);
  const imgT = document.createElement("img");
  imgT.src = rec.thumb; imgT.loading = "lazy"; imgT.alt = rec.id; imgT.title = rec.id;
  const imgC = document.createElement("img");
  imgC.src = rec.cam; imgC.loading = "lazy";
  const orig = document.createElement("a");
  orig.href = "file:///" + rec.path.replace(/\\\\/g, "/");
  orig.target = "_blank";
  orig.textContent = "open"; orig.style.position = "absolute"; orig.style.right = "6px";
  orig.style.top = "6px"; orig.style.background = "rgba(0,0,0,.6)"; orig.style.color = "#fff";
  orig.style.fontSize = "11px"; orig.style.padding = "2px 6px"; orig.style.borderRadius = "4px";
  imgs.appendChild(imgT); imgs.appendChild(imgC); imgs.appendChild(orig);

  const body = document.createElement("div"); body.className = "body";
  const fn = document.createElement("div"); fn.className = "fn";
  fn.textContent = rec.id + " (" + rec.w + "x" + rec.h + ")";
  const r1 = document.createElement("div"); r1.className = "row";
  r1.innerHTML = `<span class="lab">GT</span><span class="gt">${rec.cls}</span>`;
  const r2 = document.createElement("div"); r2.className = "row";
  const k2 = ok ? "p-ok" : "p-bad";
  r2.innerHTML = `<span class="lab">pred</span><span class="pred ${k2}">${rec.pred}` +
    ` <span class="conf">${(rec.conf*100).toFixed(1)}%</span></span>`;
  const probs = Object.entries(rec.probs).map(([c,p])=>`${c} ${(p*100).toFixed(0)}%`).join(" · ");
  const r3 = document.createElement("div"); r3.className = "conf"; r3.textContent = probs;

  const marksRow = document.createElement("div"); marksRow.className = "marks";
  const mkBtn = (key, label, cls) => {
    const b = document.createElement("button");
    b.className = cls + (mark === key ? " on" : "");
    b.textContent = label;
    b.onclick = () => {
      const mk2 = marks[rec.id] || {};
      mk2.mark = (mark === key) ? "" : key;
      marks[rec.id] = mk2;
      save();
      refresh();
    };
    return b;
  };
  marksRow.appendChild(mkBtn("ok","✓ good","m-ok"));
  marksRow.appendChild(mkBtn("bad","✗ wrong label","m-bad"));
  marksRow.appendChild(mkBtn("maybe","? unsure","m-maybe"));

  const note = document.createElement("input");
  note.className = "note"; note.placeholder = "note...";
  note.value = mk.note || "";
  note.oninput = () => { const mk2 = marks[rec.id] || {}; mk2.note = note.value; marks[rec.id] = mk2; save(); };

  body.appendChild(fn); body.appendChild(r1); body.appendChild(r2);
  body.appendChild(r3); body.appendChild(marksRow); body.appendChild(note);
  card.appendChild(imgs); card.appendChild(body);
  return card;
}

function refresh(){
  const f = id => document.getElementById(id).value;
  const split = f("f-split"), cls = f("f-cls"), pred = f("f-pred"),
        mark = f("f-mark"), wrong = document.getElementById("f-wrong").checked,
        conf = parseFloat(f("f-conf") || 0), q = document.getElementById("search").value.toLowerCase();
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  let n = 0;
  for(const rec of DATA.rows){
    const mk = marks[rec.id] || {}; const m = mk.mark || "";
    if(split && rec.split !== split) continue;
    if(cls && rec.cls !== cls) continue;
    if(pred && rec.pred !== pred) continue;
    if(mark === "none" && m) continue;
    else if(mark && mark !== m) continue;
    if(wrong && rec.cls === rec.pred) continue;
    if(rec.conf < conf) continue;
    if(q && !rec.id.toLowerCase().includes(q)) continue;
    const card = cardClone(rec);
    card.querySelectorAll("img")[1].style.display = showCams ? "" : "none";
    grid.appendChild(card); n++;
  }
  document.getElementById("empty").style.display = n ? "none" : "";
  document.getElementById("stats").innerHTML = statsHTML();
}

["f-split","f-cls","f-pred","f-mark","f-wrong","f-conf","search"].forEach(id =>
  document.getElementById(id).addEventListener("input", refresh));
document.addEventListener("change", e => { if(e.target.id === "f-wrong") refresh(); });

document.getElementById("toggle-all").onclick = () => { showCams = !showCams; refresh(); };
document.getElementById("clear-marks").onclick = () => {
  if(confirm("Clear all marks in this browser?")){ marks = {}; save(); refresh(); }
};
document.getElementById("exp").onclick = () => {
  const blob = new Blob([JSON.stringify({dataset: DATA.checkpoint, marks}, null, 2)],
    {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gallery_marks.json"; a.click();
};
document.getElementById("imp-btn").onclick = () => document.getElementById("imp").click();
document.getElementById("imp").onchange = e => {
  const f = e.target.files[0]; if(!f) return;
  const rd = new FileReader();
  rd.onload = () => { try{ const j = JSON.parse(rd.result); marks = Object.assign(marks, j.marks||{}); save(); refresh(); }catch(_){ alert("bad JSON") } };
  rd.readAsText(f); e.target.value = "";
};

refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()