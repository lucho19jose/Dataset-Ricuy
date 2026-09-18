# -*- coding: utf-8 -*-
"""Evaluación R1: valida un modelo sobre un split y guarda métricas globales, por clase
y las estadísticas por imagen (para el bootstrap pareado por grupos).

Uso:
    python r1_eval.py --weights ../best.pt --data dataset_v2/data_E2.yaml --split test --imgsz 1024 --name E0_1024
    python r1_eval.py --bootstrap A.npz B.npz      # IC 95 % de la diferencia B - A (mAP50 caja y máscara)
"""
import os, sys, json, argparse, csv, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "runs", "r1", "eval")
MANIFEST = os.path.join(HERE, "dataset_v2", "split_manifest.csv")
_PER_IMAGE = []  # se llena durante model.val()
_CONF = {}       # matriz de confusión (conf 0.25, IoU 0.45 de Ultralytics) del último model.val()


def _validator_class():
    from ultralytics.models.yolo.segment import SegmentationValidator

    class RecordingValidator(SegmentationValidator):
        """Igual al validador de Ultralytics, pero conserva las estadísticas de cada imagen."""

        def init_metrics(self, model):
            super().init_metrics(model)
            _PER_IMAGE.clear()
            orig = self.metrics.update_stats

            def record(stat):
                _PER_IMAGE.append({k: np.asarray(stat[k]) if k != "im_name" else stat[k]
                                   for k in ("tp", "tp_m", "conf", "pred_cls", "target_cls", "im_name")})
                orig(stat)

            self.metrics.update_stats = record

        def finalize_metrics(self):
            super().finalize_metrics()
            cm = getattr(self, "confusion_matrix", None)
            _CONF["matrix"] = None if cm is None else np.array(cm.matrix).copy()

    return RecordingValidator


def evaluate(weights, data, split="test", imgsz=640, name=None, batch=4, plots=True):
    from ultralytics import YOLO
    name = name or os.path.splitext(os.path.basename(weights))[0]
    os.makedirs(EVAL_DIR, exist_ok=True)
    model = YOLO(weights)
    m = model.val(validator=_validator_class(), data=data, split=split, imgsz=imgsz, batch=batch,
                  workers=0, device=0, plots=plots, verbose=False, project=EVAL_DIR,
                  name=f"{name}_{split}", exist_ok=True)
    names = m.names
    per_class = {}
    for j, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = dict(
            box50=round(float(m.box.ap50[j]), 4), box5095=round(float(m.box.ap[j]), 4),
            mask50=round(float(m.seg.ap50[j]), 4), mask5095=round(float(m.seg.ap[j]), 4),
            boxP=round(float(m.box.p[j]), 4), boxR=round(float(m.box.r[j]), 4))
    res = dict(weights=os.path.relpath(weights, HERE), data=os.path.relpath(data, HERE), split=split,
               imgsz=imgsz, n_images=len(_PER_IMAGE),
               overall=dict(box50=round(float(m.box.map50), 4), box5095=round(float(m.box.map), 4),
                            boxP=round(float(m.box.mp), 4), boxR=round(float(m.box.mr), 4),
                            mask50=round(float(m.seg.map50), 4), mask5095=round(float(m.seg.map), 4),
                            maskP=round(float(m.seg.mp), 4), maskR=round(float(m.seg.mr), 4)),
               speed_ms=m.speed, per_class=per_class)
    base = os.path.join(EVAL_DIR, f"{name}_{split}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(base + ".pkl", "wb") as f:
        pickle.dump(list(_PER_IMAGE), f)
    if _CONF.get("matrix") is not None:
        np.save(base + "_confusion.npy", _CONF["matrix"])
    # control: el mAP recalculado desde las estadísticas por imagen debe coincidir con Ultralytics
    chk = map_from_stats(_PER_IMAGE)
    print(f"[{name}/{split}@{imgsz}] box50={res['overall']['box50']} mask50={res['overall']['mask50']} "
          f"| recalculado box50={chk['box50']:.4f} mask50={chk['mask50']:.4f}")
    return res


def map_from_stats(per_image, subset=None):
    """mAP50 y mAP50-95 (caja y máscara) con ap_per_class de Ultralytics sobre un subconjunto de imágenes."""
    from ultralytics.utils.metrics import ap_per_class
    rows = per_image if subset is None else [per_image[i] for i in subset]
    cat = lambda k: np.concatenate([r[k] for r in rows], 0) if rows else np.zeros(0)
    out = {}
    tcls = cat("target_cls")
    for key, tp_key in (("box", "tp"), ("mask", "tp_m")):
        tp = np.concatenate([r[tp_key].reshape(-1, 10) for r in rows], 0)
        if tp.shape[0] == 0 or tcls.size == 0:
            out[key + "50"], out[key + "5095"], out[key + "_ap50_cls"] = 0.0, 0.0, {}
            continue
        r = ap_per_class(tp, cat("conf"), cat("pred_cls"), tcls)
        ap, cls_idx = r[5], r[6]  # ap (nc x 10), clases presentes
        out[key + "50"] = float(ap[:, 0].mean())
        out[key + "5095"] = float(ap.mean())
        out[key + "_ap50_cls"] = {int(c): float(a) for c, a in zip(cls_idx, ap[:, 0])}
    return out


def load_groups():
    g = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g[row["archivo"]] = (row["grupo"], row["fuente"])
    return g


def bootstrap(pkl_a, pkl_b, n=2000, seed=0, exclude_sv=True):
    """IC 95 % de B - A remuestreando GRUPOS del test (pareado: mismas imágenes en ambos)."""
    A = pickle.load(open(pkl_a, "rb"))
    B = pickle.load(open(pkl_b, "rb"))
    groups = load_groups()
    idx_a = {r["im_name"]: i for i, r in enumerate(A)}
    idx_b = {r["im_name"]: i for i, r in enumerate(B)}
    names = [k for k in idx_a if k in idx_b and not (exclude_sv and groups[k][1] == "SV")]
    by_g = {}
    for k in names:
        by_g.setdefault(groups[k][0], []).append(k)
    G = sorted(by_g)
    rng = np.random.default_rng(seed)
    point_a = map_from_stats(A, [idx_a[k] for k in names])
    point_b = map_from_stats(B, [idx_b[k] for k in names])
    diffs = {"box50": [], "mask50": []}
    for _ in range(n):
        pick = rng.choice(len(G), len(G), replace=True)
        ks = [k for gi in pick for k in by_g[G[gi]]]
        ma = map_from_stats(A, [idx_a[k] for k in ks])
        mb = map_from_stats(B, [idx_b[k] for k in ks])
        for key in diffs:
            diffs[key].append(mb[key] - ma[key])
    res = {"n_imagenes": len(names), "n_grupos": len(G), "excluye_street_view": exclude_sv}
    for key, v in diffs.items():
        v = np.asarray(v)
        res[key] = dict(A=round(point_a[key], 4), B=round(point_b[key], 4),
                        diff=round(point_b[key] - point_a[key], 4),
                        ic95=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name")
    ap.add_argument("--bootstrap", nargs=2)
    a = ap.parse_args()
    if a.bootstrap:
        print(json.dumps(bootstrap(*a.bootstrap), indent=2))
    else:
        evaluate(a.weights, a.data, a.split, a.imgsz, a.name)
