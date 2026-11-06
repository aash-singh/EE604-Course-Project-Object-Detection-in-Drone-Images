# %% [markdown]
# # Visual Object Search in Drone Images
# **Query-by-example, training-free search with DINOv2 patch features**
#
# Input : a drone image + a query (a box drawn on a support image showing the object of interest)
# Output: boxes (with scores and estimated orientation) of every instance of that object in the image
#
# **Kaggle setup (do this first)**
# 1. Settings -> Accelerator -> GPU (T4 or P100)
# 2. Settings -> Internet -> ON (needed to download DINOv2 / OWL-ViT weights)
# 3. Add Input -> search "VisDrone" -> add a dataset containing VisDrone2019-DET (train + val, with images/ and annotations/ folders)
#
# **Prior work this builds on (cite in the report)**
# * DINOv2 backbone: Oquab et al., 2023 (arXiv 2304.07193)
# * Tiling of large aerial images: standard practice, e.g. SAHI (Akyon et al., 2022) and Xie et al. (arXiv 2212.13136)
# * Reference-based detection with DINOv2 features on remote-sensing images: Bou et al., 2024 (arXiv 2403.05381)
# * Training-free DINOv2 patch matching with rotated references: AnomalyDINO, Damm et al. (arXiv 2405.14529)
# * Training-free reference-based segmentation with DINOv2 + SAM2: Espinosa et al., 2025 (arXiv 2507.02798)
# * Baseline: OWL-ViT image-guided detection, Minderer et al., ECCV 2022
# * Data: VisDrone, Zhu et al., TPAMI 2021
#
# **Our contributions (what is ours vs. borrowed)**
# 1. Query-scale-adaptive pyramid: the target scale levels are derived from the pixel size of the query box, instead of a fixed pyramid.
# 2. Orientation-aware matching: the *support image* is rotated (0/90/180/270) and re-embedded (ViT features are not rotation-equivariant); the best rotation is returned as the object's estimated orientation.
# 3. Coarse-to-fine search: refine only the top candidates at 2x resolution in a local window (sub-patch localisation + re-ranking).
# 4. Systematic ablation of each component on VisDrone against fixed-scale tiling and OWL-ViT.

# %% Imports and global settings
import os, time, json, math, random, glob
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.ops import nms, roi_align
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PATCH = 14                       # DINOv2 patch size
OUT_DIR = "/kaggle/working/results" if os.path.exists("/kaggle") else "./results"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
print("device:", DEVICE)


def sync():
    """Wait for the GPU so that timings are honest."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# %% [markdown]
# ## 1. VisDrone loading
# VisDrone-DET annotation line: `x,y,w,h,score,category,truncation,occlusion`.
# * category 0 = *ignored region*; 1..10 = pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
# * score = 1 -> counted in evaluation, 0 -> ignored

# %% VisDrone helpers
CLASSES = {1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
           6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor"}


def find_split_dir(root, key):
    """Find a folder whose name contains `key` ('train'/'val') and that has images/ and annotations/."""
    for dp, dn, _ in os.walk(root):
        if key in os.path.basename(dp).lower() and "images" in dn and "annotations" in dn:
            return dp
    return None


def read_ann(path):
    """Return an (N,8) int array: x,y,w,h,score,cat,trunc,occ."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 8:
                rows.append([int(float(v)) for v in p[:8]])
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def img_path_of(split_dir, ann_path):
    stem = os.path.splitext(os.path.basename(ann_path))[0]
    return os.path.join(split_dir, "images", stem + ".jpg")


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


# %% [markdown]
# ## 2. Backbone and dense feature extraction (tiling)
# A large image is split into 518x518 tiles (37x37 patches) with 25% overlap; token grids of overlapping tiles are averaged
# and stitched into one dense, L2-normalised patch-feature map for the whole image.

# %% Backbone
class DinoBackbone:
    """Frozen DINOv2 (HuggingFace). tokens(x) maps (B,3,H,W) in [0,1] -> (B,C,H/14,W/14)."""

    def __init__(self, name="facebook/dinov2-base"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name).to(DEVICE).eval()
        if DEVICE == "cuda":
            self.model.half()
        self.dtype = next(self.model.parameters()).dtype
        self.dim = self.model.config.hidden_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    @torch.no_grad()
    def tokens(self, x):
        x = ((x - self.mean) / self.std).to(self.dtype)
        out = self.model(pixel_values=x).last_hidden_state[:, 1:]      # drop CLS token
        B, N, C = out.shape
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        return out.reshape(B, h, w, C).permute(0, 3, 1, 2).float()


def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255).to(DEVICE)


def resize_to_patch_multiple(x, scale):
    """Resize (3,H,W) by ~scale so both sides are multiples of 14. Returns (tensor, sx, sy) with sx = newW/W."""
    _, H, W = x.shape
    nh = max(PATCH, int(round(H * scale / PATCH)) * PATCH)
    nw = max(PATCH, int(round(W * scale / PATCH)) * PATCH)
    shrink = (nh < H) or (nw < W)
    y = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=shrink)[0]
    return y, nw / W, nh / H


def _starts(n, t, st):
    """Tile start indices covering a length-n axis with tiles of size t and stride st."""
    if n <= t:
        return [0], n
    s = list(range(0, n - t + 1, st))
    if s[-1] + t < n:
        s.append(n - t)
    return s, t


@torch.no_grad()
def dense_features(bb, img, tile=518, stride=364, bs=8):
    """Tiled DINOv2 features for a (3,H,W) image (H,W multiples of 14) -> normalised (C,h,w) map."""
    _, H, W = img.shape
    h, w = H // PATCH, W // PATCH
    t, st = tile // PATCH, stride // PATCH
    ys, th = _starts(h, t, st)
    xs, tw = _starts(w, t, st)
    acc = torch.zeros(bb.dim, h, w, device=img.device)
    cnt = torch.zeros(1, h, w, device=img.device)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), bs):
        chunk = coords[i:i + bs]
        batch = torch.stack([img[:, y * PATCH:(y + th) * PATCH, x * PATCH:(x + tw) * PATCH] for y, x in chunk])
        tok = bb.tokens(batch)
        for (y, x), tk in zip(chunk, tok):
            acc[:, y:y + th, x:x + tw] += tk
            cnt[:, y:y + th, x:x + tw] += 1
    return F.normalize(acc / cnt, dim=0)


class TargetFeatures:
    """Lazily computed, cached feature pyramid of ONE target image.
    Level k has scale 2**(k/2) (half-octave steps). Only requested levels are computed."""

    def __init__(self, bb, pil):
        self.bb = bb
        self.img = pil_to_tensor(pil)
        _, self.H, self.W = self.img.shape
        self.cache, self.cost = {}, {}

    def level(self, k, max_side=4032):
        if k in self.cache:
            return self.cache[k]
        s = 2 ** (k / 2)
        if max(self.H, self.W) * s > max_side or min(self.H, self.W) * s < 3 * PATCH:
            self.cache[k] = None
            return None
        sync(); t0 = time.time()
        x, sx, sy = resize_to_patch_multiple(self.img, s)
        feat = dense_features(self.bb, x)
        sync()
        self.cost[k] = time.time() - t0
        self.cache[k] = dict(feat=feat.to(FEAT_DTYPE), sx=sx, sy=sy)
        return self.cache[k]


# %% [markdown]
# ## 3. Query encoding (orientation-aware)
# The query is a box on a support image. We crop a context window around it, rescale so the object spans about `n_p` patches,
# **rotate** the window by 0/90/180/270 degrees, embed it, and ROI-pool the features inside the box into a small
# `gh x gw` kernel of patch descriptors. Rotating the *image* (not the feature map) is required because ViT features are not rotation-equivariant.

# %% Query encoding
@torch.no_grad()
def extract_query(bb, support, box, rq, rot, min_ctx_px=224, ctx=1.5):
    """support: (3,H,W) tensor; box=(x,y,w,h) in support px; rq: rescale factor; rot: number of 90-degree rotations.
    Returns dict(K=(C,gh,gw) normalised kernel, wh=(w,h) of the box in rescaled+rotated px, rot)."""
    _, H, W = support.shape
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + w / 2, y + h / 2
    half = max(max(w, h) * ctx, (min_ctx_px / 2) / rq)             # context window half-size (original px)
    x0, x1 = int(max(0, math.floor(cx - half))), int(min(W, math.ceil(cx + half)))
    y0, y1 = int(max(0, math.floor(cy - half))), int(min(H, math.ceil(cy + half)))
    crop = support[:, y0:y1, x0:x1]
    mask = torch.zeros(1, y1 - y0, x1 - x0, device=support.device)
    bx0, bx1 = int(max(0, round(x - x0))), int(min(x1 - x0, round(x + w - x0)))
    by0, by1 = int(max(0, round(y - y0))), int(min(y1 - y0, round(y + h - y0)))
    mask[:, by0:max(by1, by0 + 1), bx0:max(bx1, bx0 + 1)] = 1
    crop_s, _, _ = resize_to_patch_multiple(crop, rq)
    mask_s = F.interpolate(mask[None], size=crop_s.shape[-2:], mode="nearest")[0]
    crop_s = torch.rot90(crop_s, rot, dims=(1, 2))
    mask_s = torch.rot90(mask_s, rot, dims=(1, 2))
    tok = bb.tokens(crop_s[None])[0]                                 # (C,h,w)
    ys, xs = torch.nonzero(mask_s[0], as_tuple=True)
    mx0, mx1 = xs.min().item(), xs.max().item() + 1
    my0, my1 = ys.min().item(), ys.max().item() + 1
    gh = max(1, int(round((my1 - my0) / PATCH)))
    gw = max(1, int(round((mx1 - mx0) / PATCH)))
    rois = torch.tensor([[0, mx0 / PATCH, my0 / PATCH, mx1 / PATCH, my1 / PATCH]],
                        device=tok.device, dtype=tok.dtype)
    K = roi_align(tok[None], rois, output_size=(gh, gw), spatial_scale=1.0, sampling_ratio=2, aligned=True)[0]
    K = F.normalize(K, dim=0)                                        # per-cell L2 normalisation
    return dict(K=K.to(FEAT_DTYPE), wh=(mx1 - mx0, my1 - my0), rot=rot)


# %% [markdown]
# ## 4. Search: query-scale-adaptive pyramid + rotation-aware template matching + coarse-to-fine refinement

# %% Search
@dataclass
class SearchConfig:
    name: str = "full"
    n_p: int = 3                 # object size (in ViT patches) after rescaling the query
    mode: str = "adaptive"       # "single" | "fixed" | "adaptive"  (target-scale policy)
    span: int = 2                # half-octave levels on each side of the centre level
    rotations: tuple = (0, 1, 2, 3)
    refine: bool = True          # coarse-to-fine
    min_score: float = 0.25      # minimum mean cosine similarity for a peak
    topk_per_map: int = 300
    nms_iou: float = 0.5
    max_dets: int = 300
    refine_top: int = 150        # how many candidates get refined
    alpha: float = 0.5           # final = alpha*coarse + (1-alpha)*fine
    max_side: int = 4032         # largest allowed pyramid-level side (px)


def choose_levels(cfg, rq):
    if cfg.mode == "single":
        return [0]                                              # native resolution only
    if cfg.mode == "fixed":
        return list(range(-cfg.span, cfg.span + 1))            # same levels for every query
    k0 = int(round(2 * math.log2(rq)))                         # level where the query object has n_p patches
    return list(range(k0 - cfg.span, k0 + cfg.span + 1))       # adaptive: centred on the query scale


def match_map(feat, K):
    """Mean cosine similarity of kernel K (C,gh,gw) at every position of feat (C,h,w). Output (h-gh+1, w-gw+1)."""
    gh, gw = K.shape[1:]
    if feat.shape[1] < gh or feat.shape[2] < gw:
        return None
    return F.conv2d(feat[None], (K / (gh * gw))[None])[0, 0].float()


@torch.no_grad()
def coarse_candidates(tf, kernels, levels, cfg):
    """Slide every rotated kernel over every pyramid level; keep local maxima; map to original-image boxes."""
    B, S, KS, RS = [], [], [], []
    for k in levels:
        L = tf.level(k, cfg.max_side)
        if L is None:
            continue
        for q in kernels:
            sm = match_map(L["feat"], q["K"])
            if sm is None:
                continue
            gh, gw = q["K"].shape[1:]
            mx = F.max_pool2d(sm[None, None], 3, 1, 1)[0, 0]
            ys, xs = torch.nonzero((sm == mx) & (sm > cfg.min_score), as_tuple=True)
            if len(ys) == 0:
                continue
            sc = sm[ys, xs]
            if len(sc) > cfg.topk_per_map:
                top = sc.topk(cfg.topk_per_map).indices
                ys, xs, sc = ys[top], xs[top], sc[top]
            cxl = (xs.float() + gw / 2) * PATCH                  # centre in level px
            cyl = (ys.float() + gh / 2) * PATCH
            wq, hq = q["wh"]
            B.append(torch.stack([(cxl - wq / 2) / L["sx"], (cyl - hq / 2) / L["sy"],
                                  (cxl + wq / 2) / L["sx"], (cyl + hq / 2) / L["sy"]], 1))
            S.append(sc)
            KS.append(torch.full_like(sc, k, dtype=torch.long))
            RS.append(torch.full_like(sc, q["rot"], dtype=torch.long))
    if not B:
        return None
    boxes, scores, ks, rots = torch.cat(B), torch.cat(S), torch.cat(KS), torch.cat(RS)
    keep = nms(boxes, scores, cfg.nms_iou)[:cfg.max_dets]
    return dict(boxes=boxes[keep], scores=scores[keep], k=ks[keep], rot=rots[keep])


@torch.no_grad()
def refine_candidates(bb, tf, support, qbox, rq, dets, cfg):
    """Coarse-to-fine: re-embed a local window around each top candidate at 2x its level resolution,
    re-match with a 2x-resolution query kernel inside +-2 fine patches of the coarse centre, and re-score."""
    fine_q = {r: extract_query(bb, support, qbox, 2 * rq, r) for r in set(dets["rot"].tolist())}
    boxes, scores = dets["boxes"].clone(), dets["scores"].clone()
    new_scores = scores * cfg.alpha                              # candidates that are not refined are demoted
    m = min(len(scores), cfg.refine_top)
    for i in range(m):
        k, r = int(dets["k"][i]), int(dets["rot"][i])
        s_f = 2 * (2 ** (k / 2))                                 # fine scale = 2x the coarse level scale
        x0, y0, x1, y1 = boxes[i].tolist()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(max(x1 - x0, y1 - y0) * 1.5, 64 / s_f)        # window ~3x the object (at least ~128 px at fine scale)
        wx0, wx1 = int(max(0, math.floor(cx - half))), int(min(tf.W, math.ceil(cx + half)))
        wy0, wy1 = int(max(0, math.floor(cy - half))), int(min(tf.H, math.ceil(cy + half)))
        if wx1 - wx0 < 4 or wy1 - wy0 < 4:
            continue
        win_s, sx, sy = resize_to_patch_multiple(tf.img[:, wy0:wy1, wx0:wx1], s_f)
        feat = F.normalize(bb.tokens(win_s[None])[0], dim=0).to(FEAT_DTYPE)
        K = fine_q[r]["K"]
        sm = match_map(feat, K)
        if sm is None:
            continue
        gh, gw = K.shape[1:]
        # restrict to positions near the coarse centre (avoids jumping to a neighbouring instance)
        ecx, ecy = (cx - wx0) * sx, (cy - wy0) * sy
        px = (torch.arange(sm.shape[1], device=sm.device).float() + gw / 2) * PATCH
        py = (torch.arange(sm.shape[0], device=sm.device).float() + gh / 2) * PATCH
        near = ((py[:, None] - ecy).abs() <= 2 * PATCH) & ((px[None, :] - ecx).abs() <= 2 * PATCH)
        if not near.any():
            continue
        sm = sm.masked_fill(~near, -1.0)
        idx = int(sm.argmax())
        yy, xx = divmod(idx, sm.shape[1])
        fs = float(sm[yy, xx])
        cxl, cyl = (xx + gw / 2) * PATCH, (yy + gh / 2) * PATCH
        wq, hq = fine_q[r]["wh"]
        boxes[i] = torch.tensor([wx0 + (cxl - wq / 2) / sx, wy0 + (cyl - hq / 2) / sy,
                                 wx0 + (cxl + wq / 2) / sx, wy0 + (cyl + hq / 2) / sy], device=boxes.device)
        new_scores[i] = cfg.alpha * scores[i] + (1 - cfg.alpha) * fs
    keep = nms(boxes, new_scores, cfg.nms_iou)[:cfg.max_dets]
    return dict(boxes=boxes[keep], scores=new_scores[keep], k=dets["k"][keep], rot=dets["rot"][keep])


@torch.no_grad()
def run_search(bb, tf, support, qbox, cfg):
    """Full pipeline for one query on one target image. Returns dict(boxes, scores, k, rot, time)."""
    before = set(tf.cost.keys())
    sync(); t0 = time.time()
    w, h = qbox[2], qbox[3]
    rq = cfg.n_p * PATCH / max(w, h)                             # rescale so the query object spans ~n_p patches
    kernels = [extract_query(bb, support, qbox, rq, r) for r in cfg.rotations]
    levels = choose_levels(cfg, rq)
    dets = coarse_candidates(tf, kernels, levels, cfg)
    if dets is not None and cfg.refine:
        dets = refine_candidates(bb, tf, support, qbox, rq, dets, cfg)
    sync()
    wall = time.time() - t0
    # fair timing: add the cost of levels that were already cached by an earlier configuration
    reused = sum(tf.cost[k] for k in levels if k in tf.cost and k in before)
    if dets is None:
        return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), k=np.zeros(0), rot=np.zeros(0), time=wall + reused)
    return dict(boxes=dets["boxes"].cpu().numpy(), scores=dets["scores"].cpu().numpy(),
                k=dets["k"].cpu().numpy(), rot=dets["rot"].cpu().numpy(), time=wall + reused)


# %% [markdown]
# ## 5. Evaluation (episodic one-shot protocol)
# * **Episode** = (target image from VisDrone *val*, category c, query = one clean instance of c from a *different* VisDrone *train* image).
# * Task: find all instances of c in the target image. Ground truth = category-c boxes with score=1; detections inside "ignored regions" are dropped.
# * Metrics: **AP@0.5** per episode (all-point interpolation), **Recall@100**, and **runtime** per episode. We report the mean over episodes, per config, per class, and per object-size bucket.
# * All configurations see exactly the same episodes (fixed seed).

# %% Metrics (pure numpy)
def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def eval_episode(det_boxes, det_scores, gt_xyxy, ignore_xyxy, thr=0.5, K=100):
    """Return (AP, Recall@K). det_boxes xyxy; gt/ignore xyxy."""
    n_gt = len(gt_xyxy)
    if n_gt == 0:
        return None
    det_boxes = np.asarray(det_boxes, float).reshape(-1, 4)
    det_scores = np.asarray(det_scores, float).reshape(-1)
    if len(ignore_xyxy) and len(det_boxes):                       # drop detections centred in ignored regions
        c = np.stack([(det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2], 1)
        inside = np.zeros(len(det_boxes), bool)
        for g in ignore_xyxy:
            inside |= (c[:, 0] >= g[0]) & (c[:, 0] <= g[2]) & (c[:, 1] >= g[1]) & (c[:, 1] <= g[3])
        det_boxes, det_scores = det_boxes[~inside], det_scores[~inside]
    if len(det_boxes) == 0:
        return 0.0, 0.0
    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    iou = iou_matrix(det_boxes, gt_xyxy)
    matched = np.zeros(n_gt, bool)
    tp = np.zeros(len(det_boxes))
    for i in range(len(det_boxes)):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= thr and not matched[j]:
            tp[i] = 1; matched[j] = True
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):                        # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, float(tp[:K].sum() / n_gt)


# %% Episode construction
def build_episodes(train_dir, val_dir, n_images=60, cats_per_image=2, min_gt=3, seed=0):
    rng = random.Random(seed)
    pool = defaultdict(list)                                      # category -> [(support_img, box_xywh)]
    train_anns = sorted(glob.glob(os.path.join(train_dir, "annotations", "*.txt")))[::5]
    for ap_ in train_anns:
        a = read_ann(ap_)
        for x, y, w, h, sc, c, tr, oc in a:
            if c in CLASSES and sc == 1 and tr == 0 and oc == 0 and min(w, h) >= 12 and max(w, h) <= 300:
                pool[int(c)].append((img_path_of(train_dir, ap_), (int(x), int(y), int(w), int(h))))
    val_anns = sorted(glob.glob(os.path.join(val_dir, "annotations", "*.txt")))
    rng.shuffle(val_anns)
    episodes, n_img = [], 0
    for ap_ in val_anns:
        a = read_ann(ap_)
        cats = [c for c in CLASSES if ((a[:, 5] == c) & (a[:, 4] == 1)).sum() >= min_gt and len(pool[c]) > 0]
        if not cats:
            continue
        rng.shuffle(cats)
        for c in cats[:cats_per_image]:
            sup, qbox = rng.choice(pool[c])
            gt = a[(a[:, 5] == c) & (a[:, 4] == 1)]
            episodes.append(dict(ann=ap_, img=img_path_of(val_dir, ap_), cat=c, support=sup, qbox=qbox,
                                 gt_med=float(np.median(np.maximum(gt[:, 2], gt[:, 3]))), n_gt=len(gt)))
        n_img += 1
        if n_img >= n_images:
            break
    print(f"{len(episodes)} episodes on {n_img} images")
    return episodes


# %% [markdown]
# ## 6. OWL-ViT image-guided baseline (tiled)
# Baseline from Minderer et al. (ECCV 2022): image-conditioned detection. Run on 768x768 tiles because the model resizes its input to 768x768.

# %% OWL-ViT baseline
class OwlBaseline:
    def __init__(self, name="google/owlvit-base-patch32"):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        self.proc = OwlViTProcessor.from_pretrained(name)
        self.model = OwlViTForObjectDetection.from_pretrained(name).to(DEVICE).eval()

    @torch.no_grad()
    def search(self, target_pil, support_pil, qbox, scale=1.0, tile=768, stride=576, thr=0.05):
        x, y, w, h = qbox
        m = 0.15 * max(w, h)                                      # small context margin around the query crop
        query = support_pil.crop((max(0, x - m), max(0, y - m), x + w + m, y + h + m))
        img = target_pil if scale == 1.0 else target_pil.resize(
            (int(target_pil.width * scale), int(target_pil.height * scale)), Image.BILINEAR)
        W, H = img.size
        xs, tw = _starts(W, tile, stride)
        ys, th = _starts(H, tile, stride)
        sync(); t0 = time.time()
        B, S = [], []
        for ty in ys:
            for tx in xs:
                t_img = img.crop((tx, ty, tx + tw, ty + th))
                inp = self.proc(images=t_img, query_images=query, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                out = self.model.image_guided_detection(**inp)
                res = self.proc.post_process_image_guided_detection(
                    outputs=out, threshold=thr, nms_threshold=0.3,
                    target_sizes=torch.tensor([[th, tw]], device=out.logits.device))[0]
                b = res["boxes"].cpu().numpy() + np.array([tx, ty, tx, ty])
                B.append(b / scale); S.append(res["scores"].cpu().numpy())
        sync()
        if not B or sum(len(s) for s in S) == 0:
            return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), time=time.time() - t0)
        boxes, scores = np.concatenate(B), np.concatenate(S)
        keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32), 0.5)[:300].numpy()
        return dict(boxes=boxes[keep], scores=scores[keep], time=time.time() - t0)


# %% [markdown]
# ## 7. Run the benchmark

# %% Benchmark runner
def run_benchmark(bb, episodes, configs, owl=None, owl_scales=(1.0, 2.0)):
    records = []
    by_img = defaultdict(list)
    for ep in episodes:
        by_img[ep["img"]].append(ep)
    owl_ok = owl is not None
    for n, (img_path, eps) in enumerate(by_img.items()):
        pil = Image.open(img_path).convert("RGB")
        tf = TargetFeatures(bb, pil)                              # levels are shared (cached) across configs
        for ep in eps:
            a = read_ann(ep["ann"])
            gt = xywh_to_xyxy(a[(a[:, 5] == ep["cat"]) & (a[:, 4] == 1), :4])
            ign = xywh_to_xyxy(a[a[:, 5] == 0, :4])
            sup_pil = Image.open(ep["support"]).convert("RGB")
            sup_t = pil_to_tensor(sup_pil)
            base = dict(img=os.path.basename(img_path), cat=CLASSES[ep["cat"]], n_gt=ep["n_gt"], gt_med=ep["gt_med"])
            for cfg in configs:
                r = run_search(bb, tf, sup_t, ep["qbox"], cfg)
                m = eval_episode(r["boxes"], r["scores"], gt, ign)
                records.append(dict(base, method=cfg.name, ap=m[0], r100=m[1], time=r["time"]))
            if owl_ok:
                try:
                    for s in owl_scales:
                        r = owl.search(pil, sup_pil, ep["qbox"], scale=s)
                        m = eval_episode(r["boxes"], r["scores"], gt, ign)
                        records.append(dict(base, method=f"OWL-ViT tiled ({s:g}x)", ap=m[0], r100=m[1], time=r["time"]))
                except Exception as e:                            # API differences should not kill the whole run
                    print("OWL-ViT baseline disabled:", repr(e)); owl_ok = False
        del tf
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if (n + 1) % 5 == 0:
            print(f"{n + 1}/{len(by_img)} images done")
    return pd.DataFrame(records)


def summarize(df):
    """Main table, per-size and per-class tables (values in %, runtime in s)."""
    order = list(dict.fromkeys(df["method"]))
    main = df.groupby("method").agg(mAP50=("ap", "mean"), R100=("r100", "mean"), time_s=("time", "mean")).reindex(order)
    main[["mAP50", "R100"]] *= 100
    df = df.copy()
    df["size"] = pd.cut(df["gt_med"], [0, 32, 64, 1e9], labels=["small (<32px)", "medium (32-64px)", "large (>64px)"])
    by_size = (df.pivot_table(index="method", columns="size", values="ap", aggfunc="mean", observed=False) * 100).reindex(order)
    by_cls = (df.pivot_table(index="method", columns="cat", values="ap", aggfunc="mean") * 100).reindex(order)
    return main.round(2), by_size.round(2), by_cls.round(2)


# %% Smoke test (SMALL: 6 images) -- run this first, then increase N_IMAGES
ROOT = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
TRAIN_DIR, VAL_DIR = find_split_dir(ROOT, "train"), find_split_dir(ROOT, "val")
print("train:", TRAIN_DIR, "\nval:", VAL_DIR)
assert TRAIN_DIR and VAL_DIR, "Could not find VisDrone train/val folders (each needs images/ and annotations/). Check the Add Input dataset."

bb = DinoBackbone("facebook/dinov2-base")

CONFIGS = [
    SearchConfig(name="A: single scale",           mode="single",   rotations=(0,),          refine=False),
    SearchConfig(name="B: fixed pyramid",          mode="fixed",    rotations=(0,),          refine=False),
    SearchConfig(name="C: + adaptive scales",      mode="adaptive", rotations=(0,),          refine=False),
    SearchConfig(name="D: + rotation",             mode="adaptive", rotations=(0, 1, 2, 3),  refine=False),
    SearchConfig(name="E: + coarse-to-fine (full)", mode="adaptive", rotations=(0, 1, 2, 3), refine=True),
]
episodes = build_episodes(TRAIN_DIR, VAL_DIR, n_images=6)
df_smoke = run_benchmark(bb, episodes, CONFIGS, owl=None)
print(summarize(df_smoke)[0])

# %% Full benchmark (increase N_IMAGES as far as your Kaggle time allows; 60 images ~ 100 episodes)
N_IMAGES = 60
episodes = build_episodes(TRAIN_DIR, VAL_DIR, n_images=N_IMAGES)
owl = None
try:
    owl = OwlBaseline()
except Exception as e:
    print("OWL-ViT could not be loaded:", repr(e))
df = run_benchmark(bb, episodes, CONFIGS, owl=owl)
df.to_csv(f"{OUT_DIR}/all_episodes.csv", index=False)
main, by_size, by_cls = summarize(df)
print(main, "\n", by_size, "\n", by_cls)
for name, t in [("main", main), ("by_size", by_size), ("by_class", by_cls)]:
    t.to_csv(f"{OUT_DIR}/{name}.csv")
    open(f"{OUT_DIR}/{name}.tex", "w").write(t.to_latex(float_format="%.2f"))   # paste into the CVPR template

# %% Sensitivity to the query-object size n_p (hyper-parameter study for the report)
rows = []
for n_p in (2, 3, 4):
    cfg = SearchConfig(name=f"full, n_p={n_p}", n_p=n_p)
    d = run_benchmark(bb, episodes[:40], [cfg], owl=None)
    rows.append(summarize(d)[0])
sens = pd.concat(rows); print(sens); sens.to_csv(f"{OUT_DIR}/sensitivity_np.csv")

# %% [markdown]
# ## 8. Qualitative results (figures for the report)

# %% Visualisation
def draw(ax, pil, boxes, scores, gt=None, topn=40, title=""):
    ax.imshow(pil); ax.set_title(title, fontsize=9); ax.axis("off")
    if gt is not None:
        for b in gt:
            ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1))
    for b, s in list(zip(boxes, scores))[:topn]:
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.2))


def qualitative(bb, episodes, cfg, n=4, topn=40, save=True):
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    for row, ep in zip(axes, episodes[:n]):
        pil = Image.open(ep["img"]).convert("RGB")
        tf = TargetFeatures(bb, pil)
        sup_pil = Image.open(ep["support"]).convert("RGB")
        a = read_ann(ep["ann"])
        gt = xywh_to_xyxy(a[(a[:, 5] == ep["cat"]) & (a[:, 4] == 1), :4])
        r = run_search(bb, tf, pil_to_tensor(sup_pil), ep["qbox"], cfg)
        x, y, w, h = ep["qbox"]
        row[0].imshow(sup_pil.crop((x, y, x + w, y + h))); row[0].axis("off")
        row[0].set_title(f"query ({CLASSES[ep['cat']]})")
        draw(row[1], pil, r["boxes"], r["scores"], gt, topn, f"red=detections (top {topn}), green=GT")
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUT_DIR}/qualitative.png", dpi=130, bbox_inches="tight")
    plt.show()


qualitative(bb, episodes, CONFIGS[-1], n=4)


# %% [markdown]
# ## 9. Held-out tuning, final ablation and statistics (cells added after the first full run)

# %% Tune n_p on held-out episodes (images disjoint from the evaluation episodes)
test_imgs = {e["img"] for e in episodes}
tune_eps = [e for e in build_episodes(TRAIN_DIR, VAL_DIR, n_images=40, seed=1) if e["img"] not in test_imgs]
rows = []
for n_p in (3, 4, 5, 6):
    cfgs = [SearchConfig(name=f"fixed, n_p={n_p}", n_p=n_p, mode="fixed", rotations=(0,), refine=False),
            SearchConfig(name=f"adaptive, n_p={n_p}", n_p=n_p, mode="adaptive", rotations=(0,), refine=False),
            SearchConfig(name=f"full, n_p={n_p}", n_p=n_p)]
    rows.append(summarize(run_benchmark(bb, tune_eps, cfgs, owl=None))[0])
tune = pd.concat(rows); print(tune); tune.to_csv(f"{OUT_DIR}/tune_np.csv")

# %% Final ablation with the chosen n_p (A-E), OWL-ViT rows are re-used from the first full run
from dataclasses import replace
BEST_NP = 4   # chosen after looking at the tuning table
CONFIGS_FINAL = [replace(c, n_p=BEST_NP) for c in CONFIGS]
df_final = run_benchmark(bb, episodes, CONFIGS_FINAL, owl=None)
df_all = pd.concat([df_final, df[df.method.str.startswith("OWL")]])
df_all.to_csv(f"{OUT_DIR}/all_episodes_final.csv", index=False)
main, by_size, by_cls = summarize(df_all); print(main, by_size)

# %% Paired bootstrap confidence intervals (episode level) for consecutive ablation steps
df = pd.read_csv(f"{OUT_DIR}/all_episodes_final.csv")
print(len(df), df.method.unique())
df["ep"] = df["img"] + "|" + df["cat"]
P = df.pivot(index="ep", columns="method", values="ap") * 100
rng = np.random.default_rng(0)
def paired(a, b, n=10000):
    d = (P[a] - P[b]).dropna().values
    boots = rng.choice(d, (n, len(d))).mean(1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])
for a, b in [("B: fixed pyramid", "A: single scale"), ("C: + adaptive scales", "B: fixed pyramid"),
             ("D: + rotation", "C: + adaptive scales"), ("E: + coarse-to-fine (full)", "D: + rotation"),
             ("E: + coarse-to-fine (full)", "B: fixed pyramid")]:
    m, lo, hi = paired(a, b); print(f"{a} - {b}: {m:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

# %% Extra configurations F, G (added AFTER seeing the A-E results: exploratory) + paired CIs for AP and Recall@100
CONFIGS_EXTRA = [
    replace(SearchConfig(name="F: fixed + coarse-to-fine", mode="fixed", rotations=(0,), refine=True), n_p=BEST_NP),
    replace(SearchConfig(name="G: fixed + rotation + coarse-to-fine", mode="fixed", rotations=(0, 1, 2, 3), refine=True), n_p=BEST_NP),
]
df_extra = run_benchmark(bb, episodes, CONFIGS_EXTRA, owl=None)
df_all2 = pd.concat([df_all, df_extra])
df_all2.to_csv(f"{OUT_DIR}/all_episodes_final2.csv", index=False)
main, by_size, by_cls = summarize(df_all2); print(main); print(by_size)

d = df_all2.copy(); d["ep"] = d["img"] + "|" + d["cat"]
rng = np.random.default_rng(0)
def paired(metric, a, b, n=10000):
    P = d.pivot(index="ep", columns="method", values=metric) * 100
    x = (P[a] - P[b]).dropna().values
    boots = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(boots, [2.5, 97.5])
E, B = "E: + coarse-to-fine (full)", "B: fixed pyramid"
F_, G_ = "F: fixed + coarse-to-fine", "G: fixed + rotation + coarse-to-fine"
for metric in ("ap", "r100"):
    print("---", metric)
    for a, b in [(F_, B), (F_, E), (G_, F_), (E, B)]:
        m, lo, hi = paired(metric, a, b); print(f"{a} - {b}: {m:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

# %% Qualitative figures for the final method G (best / worst episodes by AP)
FINAL = CONFIGS_EXTRA[1]
dG = df_all2[df_all2.method == FINAL.name].sort_values("ap")
def find(r):
    return next(e for e in episodes if os.path.basename(e["img"]) == r.img and CLASSES[e["cat"]] == r.cat)
best  = [find(r) for r in dG.tail(3).itertuples()]
worst = [find(r) for r in dG.head(3).itertuples()]
qualitative(bb, best, FINAL, n=3)             # saves qualitative.png
os.replace(f"{OUT_DIR}/qualitative.png", f"{OUT_DIR}/qual_success.png")
qualitative(bb, worst, FINAL, n=3)
os.replace(f"{OUT_DIR}/qualitative.png", f"{OUT_DIR}/qual_failure.png")

# %% AP at looser IoU thresholds for the final method G (localisation diagnostic)
rows = []
by_img = defaultdict(list)
for ep in episodes: by_img[ep["img"]].append(ep)
for img_path, eps in by_img.items():
    tf = TargetFeatures(bb, Image.open(img_path).convert("RGB"))
    for ep in eps:
        a = read_ann(ep["ann"])
        gt = xywh_to_xyxy(a[(a[:, 5] == ep["cat"]) & (a[:, 4] == 1), :4])
        ign = xywh_to_xyxy(a[a[:, 5] == 0, :4])
        sup = pil_to_tensor(Image.open(ep["support"]).convert("RGB"))
        r = run_search(bb, tf, sup, ep["qbox"], FINAL)
        rows.append({f"AP@{t}": eval_episode(r["boxes"], r["scores"], gt, ign, thr=t)[0] for t in (0.1, 0.3, 0.5)})
    del tf; torch.cuda.empty_cache()
print((pd.DataFrame(rows).mean() * 100).round(2))
