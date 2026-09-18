# -*- coding: utf-8 -*-
"""Verificación controlada: ¿el salto del semáforo rojo del envío anterior (0.013 -> 0.711) era detección o
convención de anotación?

Evalúa el modelo base (QHAWAY) y el modelo del envío anterior (E1 original, runs/best_finetune_lima) sobre las
MISMAS 49 imágenes de la validación antigua, con dos versiones de etiquetas:
  (a) las originales (dataset_best/val), mezcla de cabezas y pórticos;
  (b) las armonizadas de dataset_v2 (una cabeza por polígono).
Si la brecha se cierra con (b), el salto se debía a la convención.
Uso: python r1_check_semaforo.py
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_dataset_v2 import slug
from ultralytics import YOLO

BASE = os.path.dirname(HERE)
OLD_VAL = os.path.join(HERE, "dataset_best", "images", "val")
V2 = os.path.join(HERE, "dataset_v2")
OUTD = os.path.join(HERE, "runs", "r1", "check_semaforo")
os.makedirs(OUTD, exist_ok=True)
names = sorted(os.listdir(OLD_VAL))
# lista de las mismas imágenes en dataset_v2 (prefijo de fuente + nombre normalizado)
v2imgs = []
for n in names:
    stem = os.path.splitext(n)[0]
    src = "A" if stem.startswith("video_Andahuaylas") else "L"
    p = os.path.join(V2, "images", f"{src}_{slug(stem)}.jpg")
    assert os.path.exists(p), p
    v2imgs.append(p)
lst = os.path.join(OUTD, "val_antigua_v2.txt")
open(lst, "w", encoding="utf-8").write("\n".join(v2imgs) + "\n")
yaml_v2 = os.path.join(OUTD, "val_antigua_v2.yaml")
open(yaml_v2, "w", encoding="utf-8").write(
    f"path: '{V2}'\ntrain: {lst}\nval: {lst}\nnc: 12\nnames:\n" + "".join(
        f"  {i}: {c}\n" for i, c in enumerate(["Reductor de velocidad", "Paso peatonal", "Línea recta",
        "Línea recta y derecha", "Persona", "Motocicleta", "Automóvil particular", "Camión", "Bus de transporte",
        "Semáforo en verde", "Semáforo en rojo", "Mototaxi"])))
yaml_old = os.path.join(HERE, "dataset_best", "data.yaml")
models = {"base_QHAWAY": os.path.join(BASE, "best.pt"),
          "E1_envio_anterior": os.path.join(HERE, "runs", "best_finetune_lima", "weights", "best.pt")}
res = {}
for mname, w in models.items():
    for lname, y in (("etiquetas_originales", yaml_old), ("etiquetas_armonizadas", yaml_v2)):
        m = YOLO(w).val(data=y, split="val", imgsz=640, batch=4, workers=0, device=0, plots=False, verbose=False,
                        project=OUTD, name=f"{mname}_{lname}", exist_ok=True)
        pc = {m.names[int(c)]: round(float(m.box.ap50[j]), 3) for j, c in enumerate(m.box.ap_class_index)}
        res[f"{mname}|{lname}"] = dict(box50=round(float(m.box.map50), 3), rojo=pc.get("Semáforo en rojo"),
                                       verde=pc.get("Semáforo en verde"))
        print(mname, lname, res[f"{mname}|{lname}"])
json.dump(res, open(os.path.join(OUTD, "resultado.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
