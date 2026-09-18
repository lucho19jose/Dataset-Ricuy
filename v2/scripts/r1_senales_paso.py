# -*- coding: utf-8 -*-
"""Análisis de sensibilidad (18/09): en el corpus original, parte de los polígonos de "Paso peatonal" encierran la
señal vertical amarilla de cruce peatonal y no la cebra pintada que define la clase en la prueba.

Regla: polígono de clase 1 de D1 (fuentes L y A) cuya caja es al menos tan alta como ancha en píxeles
(alto/ancho >= 0.9); las cebras y sus fragmentos son anchos (alto/ancho <= 0.4). Se verificó a ojo.

Crea dataset_v2c/ (copia de dataset_v2) con esas regiones rellenadas de gris en las imágenes y sin sus polígonos
en las etiquetas, solo en entrenamiento y validación. La prueba sigue siendo la de dataset_v2, sin cambios.
Salidas: dataset_v2c/, data_E1Rc.yaml, data_E2c.yaml y dataset_v2/senales_paso_peatonal.csv.
Uso: python r1_senales_paso.py
"""
import os, csv, json, shutil
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dataset_v2")
DST = os.path.join(HERE, "dataset_v2c")
GRIS = (114, 114, 114)

rows = list(csv.DictReader(open(os.path.join(SRC, "split_manifest.csv"), encoding="utf-8")))
senales = []
for r in rows:
    if r["fuente"] not in ("L", "A"):
        continue
    d = json.load(open(os.path.join(SRC, "unified_json", r["uid"] + ".json"), encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
    for i, s in enumerate(d["shapes"]):
        if s["label"] != "Paso peatonal":
            continue
        xs, ys = zip(*s["points"])
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if h / max(w, 1e-6) >= 0.9:
            senales.append(dict(uid=r["uid"], split=r["split"], shape_idx=i, x0=min(xs), y0=min(ys), x1=max(xs),
                                y1=max(ys), W=W, H=H, points=s["points"]))
assert all(s["split"] in ("train", "val") for s in senales), "hay señales en prueba: revisar"
with open(os.path.join(SRC, "senales_paso_peatonal.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["uid", "split", "shape_idx", "x0", "y0", "x1", "y1"])
    for s in senales:
        w.writerow([s["uid"], s["split"], s["shape_idx"]] + [round(s[k], 1) for k in ("x0", "y0", "x1", "y1")])
print(f"{len(senales)} señales de cruce anotadas como paso peatonal:",
      {sp: sum(s['split'] == sp for s in senales) for sp in ("train", "val")})

if os.path.isdir(DST):
    shutil.rmtree(DST)
os.makedirs(os.path.join(DST, "images")); os.makedirs(os.path.join(DST, "labels")); os.makedirs(os.path.join(DST, "splits"))
by_uid = {}
for s in senales:
    by_uid.setdefault(s["uid"], []).append(s)
for r in rows:
    if r["split"] == "test":
        continue
    src_img = os.path.join(SRC, "images", r["archivo"])
    dst_img = os.path.join(DST, "images", r["archivo"])
    lab = open(os.path.join(SRC, "labels", r["uid"] + ".txt"), encoding="utf-8").read().splitlines()
    if r["uid"] in by_uid:
        im = Image.open(src_img).convert("RGB")
        dr = ImageDraw.Draw(im)
        keep = []
        for line in lab:
            v = line.split()
            quitar = False
            if v and v[0] == "1":
                xy = np.array(v[1:], float).reshape(-1, 2)
                for s in by_uid[r["uid"]]:
                    bx = [s["x0"] / s["W"], s["y0"] / s["H"], s["x1"] / s["W"], s["y1"] / s["H"]]
                    if (abs(xy[:, 0].min() - bx[0]) < 0.01 and abs(xy[:, 1].min() - bx[1]) < 0.01
                            and abs(xy[:, 0].max() - bx[2]) < 0.01 and abs(xy[:, 1].max() - bx[3]) < 0.01):
                        quitar = True
            if not quitar:
                keep.append(line)
        for s in by_uid[r["uid"]]:
            m = 4
            dr.rectangle((s["x0"] - m, s["y0"] - m, s["x1"] + m, s["y1"] + m), fill=GRIS)
        assert len(lab) - len(keep) == len(by_uid[r["uid"]]), (r["uid"], len(lab), len(keep))
        im.save(dst_img, quality=95) if dst_img.lower().endswith(".jpg") else im.save(dst_img)
        lab = keep
    else:
        shutil.copy2(src_img, dst_img)
    open(os.path.join(DST, "labels", r["uid"] + ".txt"), "w", encoding="utf-8").write("\n".join(lab) + ("\n" if lab else ""))

for lst in ("E1R_train", "E1R_val", "E2_train", "E2_val"):
    src_lst = open(os.path.join(SRC, "splits", f"{lst}.txt"), encoding="utf-8").read().splitlines()
    new = [p.replace(os.path.join(SRC, "images"), os.path.join(DST, "images")) for p in src_lst if p.strip()]
    open(os.path.join(DST, "splits", f"{lst}.txt"), "w", encoding="utf-8").write("\n".join(new) + "\n")
shutil.copy2(os.path.join(SRC, "splits", "test.txt"), os.path.join(DST, "splits", "test.txt"))  # rutas a dataset_v2
base = open(os.path.join(SRC, "data_E2.yaml"), encoding="utf-8").read()
for exp, pre in (("E1Rc", "E1R"), ("E2c", "E2")):
    y = base.replace(SRC, DST).replace("splits/E2_train.txt", f"splits/{pre}_train.txt").replace(
        "splits/E2_val.txt", f"splits/{pre}_val.txt")
    open(os.path.join(DST, f"data_{exp}.yaml"), "w", encoding="utf-8").write(y)
print("->", DST)
