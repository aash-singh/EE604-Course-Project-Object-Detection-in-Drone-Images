# %% Gradio demo (paste as ONE new cell at the bottom of the notebook)
# Needs: the cells above that define DinoBackbone, TargetFeatures, SearchConfig, run_search (sections 1-4).
# Run first in a separate cell:  !pip -q install gradio
# How to use: upload a drone image, click TWO opposite corners of the object, press "Search".

import hashlib
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

import types
vc = types.SimpleNamespace(DinoBackbone=DinoBackbone, TargetFeatures=TargetFeatures, SearchConfig=SearchConfig,
                           run_search=run_search, pil_to_tensor=pil_to_tensor)
if "bb" not in globals():                      # reuse the backbone already loaded in the notebook
    bb = DinoBackbone("facebook/dinov2-base")

N_P = 4                                    # query object size in patches (value used in the report)
ROT_LABEL = {0: "0", 1: "90", 2: "180", 3: "270"}
POLICIES = {"Fixed pyramid (evaluated, config G)": "fixed",
            "Adaptive to query size (not evaluated for same-image queries)": "adaptive"}

_state = {"bb": bb, "key": None, "tf": None}


# ----------------------------------------------------------------------------- helpers (no GUI)
def get_backbone():
    """Load DINOv2 once (first call takes a while)."""
    if _state["bb"] is None:
        _state["bb"] = vc.DinoBackbone("facebook/dinov2-base")
    return _state["bb"]


def get_target(img):
    """Feature pyramid of the target image, cached so repeated queries on the same image are fast."""
    key = hashlib.md5(img.tobytes()).hexdigest() + str(img.size)
    if _state["key"] != key:
        _state["tf"] = vc.TargetFeatures(get_backbone(), img)
        _state["key"] = key
    return _state["tf"]


def click_logic(corner, x, y):
    """Two-click box selection. Returns (new_corner, box_or_None); box = (x, y, w, h)."""
    if corner is None:
        return (int(x), int(y)), None
    x0, y0 = min(corner[0], int(x)), min(corner[1], int(y))
    x1, y1 = max(corner[0], int(x)), max(corner[1], int(y))
    return None, (x0, y0, x1 - x0, y1 - y0)


def support_from_crop(crop):
    """Turn an uploaded query crop into a (support image, box) pair by padding it with its mean colour.
    Less reliable than marking the object on a real image, because the crop has no real context."""
    c = np.asarray(crop.convert("RGB"))
    h, w = c.shape[:2]
    pad = max(h, w)
    canvas = np.empty((h + 2 * pad, w + 2 * pad, 3), np.uint8)
    canvas[:] = c.reshape(-1, 3).mean(0).astype(np.uint8)
    canvas[pad:pad + h, pad:pad + w] = c
    return Image.fromarray(canvas), (pad, pad, w, h)


def merge_overlaps(boxes, scores, iou_thr=0.3, contain_thr=0.6):
    """DISPLAY-ONLY clean-up (not part of the evaluated pipeline): boxes must be sorted by score, best first.
    A box is dropped if it overlaps a kept box by IoU > iou_thr, or if it is mostly contained in / contains a kept box."""
    keep = []
    for i in range(len(boxes)):
        a, ok = boxes[i], True
        for j in keep:
            b = boxes[j]
            iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
            inter = max(iw, 0) * max(ih, 0)
            aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
            if inter / (aa + ab - inter + 1e-9) > iou_thr or inter / (min(aa, ab) + 1e-9) > contain_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def draw_boxes(img, boxes, scores, rots, query_box=None):
    """Return a copy of img with detections (red) and the query box (blue)."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, max(out.size) // 700)
    for (x0, y0, x1, y1), s, r in zip(boxes, scores, rots):
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=lw)
        d.text((x0 + 2, max(0, y0 - 11)), f"{s:.2f}/{ROT_LABEL[int(r)]}", fill=(255, 255, 0))
    if query_box is not None:
        x, y, w, h = query_box
        d.rectangle([x, y, x + w, y + h], outline=(40, 120, 255), width=lw + 1)
    return out


def preview_query(img, corner=None, box=None):
    """Show the current corner / box on a copy of the image so the user sees what was selected."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, max(out.size) // 700)
    if box is not None and box[2] > 0 and box[3] > 0:
        x, y, w, h = box
        d.rectangle([x, y, x + w, y + h], outline=(40, 120, 255), width=lw + 1)
    if corner is not None:
        r = 5 * lw
        d.ellipse([corner[0] - r, corner[1] - r, corner[0] + r, corner[1] + r], outline=(40, 120, 255), width=lw)
    return out


def do_search(img, mode, bx, by, bw, bh, crop, policy, use_rot, use_refine, min_score, max_results, clean=True):
    """Run the search. Returns (annotated image or None, table rows, status text)."""
    if img is None:
        return None, [], "Please upload a drone image first."
    img = img.convert("RGB")
    if mode == "Upload a query crop":
        if crop is None:
            return None, [], "Upload a query crop, or switch to 'Mark box on the drone image'."
        support, qbox = support_from_crop(crop)
        query_box_for_drawing = None
    else:
        qbox = (int(bx), int(by), int(bw), int(bh))
        if qbox[2] < 4 or qbox[3] < 4:
            return None, [], "The query box is too small. Click two opposite corners of the object (or set w, h >= 4)."
        W, H = img.size
        if qbox[0] < 0 or qbox[1] < 0 or qbox[0] + qbox[2] > W or qbox[1] + qbox[3] > H:
            return None, [], "The query box is outside the image."
        support, query_box_for_drawing = img, qbox

    cfg = replace(vc.SearchConfig(name="demo", n_p=N_P, mode=POLICIES[policy]),
                  rotations=(0, 1, 2, 3) if use_rot else (0,), refine=bool(use_refine))
    tf = get_target(img)
    res = vc.run_search(get_backbone(), tf, vc.pil_to_tensor(support), qbox, cfg)

    boxes, scores, rots = np.asarray(res["boxes"]), np.asarray(res["scores"]), np.asarray(res["rot"]).astype(int)
    W, H = img.size
    if len(boxes):
        boxes = np.stack([np.clip(boxes[:, 0], 0, W), np.clip(boxes[:, 1], 0, H),
                          np.clip(boxes[:, 2], 0, W), np.clip(boxes[:, 3], 0, H)], 1)
    order = np.argsort(-scores)
    order = [i for i in order if scores[i] >= min_score]
    if clean and len(order):
        order = [order[k] for k in merge_overlaps(boxes[order], scores[order])]
    order = order[:int(max_results)]
    boxes, scores, rots = boxes[order], scores[order], rots[order]

    annotated = draw_boxes(img, boxes, scores, rots, query_box_for_drawing)
    rows = [[i + 1, int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]), round(float(s), 3), ROT_LABEL[int(r)]]
            for i, (b, s, r) in enumerate(zip(boxes, scores, rots))]
    status = (f"Showing **{len(rows)}** matches (score >= {min_score:.2f}{', overlaps merged' if clean else ''}) in {res['time']:.1f} s. "
              "In 'Mark box' mode the query object itself is usually the top match.")
    return annotated, rows, status


# ----------------------------------------------------------------------------- GUI
def build_ui():
    import gradio as gr

    def on_click(img, corner, evt: gr.SelectData):
        if img is None:
            return corner, gr.update(), gr.update(), gr.update(), gr.update(), None
        x, y = evt.index[0], evt.index[1]
        corner, box = click_logic(corner, x, y)
        if box is None:   # first corner
            return corner, gr.update(), gr.update(), gr.update(), gr.update(), preview_query(img, corner=corner)
        return corner, box[0], box[1], box[2], box[3], preview_query(img, box=box)

    def on_reset(img):
        return None, 0, 0, 0, 0, (img if img is None else img.convert("RGB"))

    def on_numbers(img, bx, by, bw, bh):
        return None if img is None else preview_query(img, box=(bx, by, bw, bh))

    with gr.Blocks(title="Visual Object Search in Drone Images") as demo:
        gr.Markdown(
            "# Visual Object Search in Drone Images\n"
            "Find every instance of an object you point at, with **no training** (frozen DINOv2 + tiled feature "
            "pyramid + rotated queries + coarse-to-fine refinement).  \n"
            "**1.** Upload a drone image. **2.** Click two opposite corners of the object. **3.** Press *Search*.  \n"
            "*The first search on a new image is slower (it builds the feature pyramid). Output box size is taken "
            "from your query box, so tight query boxes work best.*")
        corner = gr.State(None)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="1) Drone image  (click two corners of the object)", height=420)
                qprev = gr.Image(label="Selected query box", interactive=False, height=300)
                with gr.Row():
                    bx = gr.Number(label="x", value=0, precision=0)
                    by = gr.Number(label="y", value=0, precision=0)
                    bw = gr.Number(label="w", value=0, precision=0)
                    bh = gr.Number(label="h", value=0, precision=0)
                reset = gr.Button("Reset box")
                mode = gr.Radio(["Mark box on the drone image", "Upload a query crop"],
                                value="Mark box on the drone image", label="Query source")
                crop = gr.Image(type="pil", label="Query crop (only used in 'Upload a query crop' mode)", height=160)
            with gr.Column():
                with gr.Accordion("Settings", open=False):
                    policy = gr.Radio(list(POLICIES), value=list(POLICIES)[0], label="Scale policy")
                    use_rot = gr.Checkbox(value=True, label="Rotation-aware query (0/90/180/270 deg)")
                    use_refine = gr.Checkbox(value=True, label="Coarse-to-fine refinement")
                    clean = gr.Checkbox(value=True, label="Merge overlapping boxes (display only; not part of the evaluated method)")
                    min_score = gr.Slider(0.2, 0.9, value=0.45, step=0.01, label="Minimum score to display")
                    max_results = gr.Slider(1, 200, value=40, step=1, label="Max matches to display")
                run = gr.Button("Search", variant="primary")
                status = gr.Markdown()
                out_img = gr.Image(label="Result (red = matches with score/rotation, blue = query)", height=520)
                table = gr.Dataframe(headers=["#", "x", "y", "w", "h", "score", "rot (deg)"], interactive=False)

        img.select(on_click, [img, corner], [corner, bx, by, bw, bh, qprev])
        img.upload(on_reset, [img], [corner, bx, by, bw, bh, qprev])
        reset.click(on_reset, [img], [corner, bx, by, bw, bh, qprev])
        for n in (bx, by, bw, bh):
            n.change(on_numbers, [img, bx, by, bw, bh], [qprev])
        run.click(do_search, [img, mode, bx, by, bw, bh, crop, policy, use_rot, use_refine, min_score, max_results, clean],
                  [out_img, table, status])
    return demo


demo = build_ui()
demo.queue().launch(share=True)
