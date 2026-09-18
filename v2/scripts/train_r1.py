# -*- coding: utf-8 -*-
"""Entrena una corrida del experimento R1 y la evalúa UNA vez en el test congelado.

Configuración común (corrige el experimento 1): parte del best.pt de QHAWAY, 640 px, batch 2 (con
batch 4 la VRAM de 4 GB se desborda a memoria compartida y cada iteración se vuelve 2.2 veces más lenta;
Ultralytics acumula gradientes hasta 64 imágenes, así que el lote efectivo no cambia),
AdamW lr0 0.000625 explícito (el que 'auto' eligió en el E1), sin volteo horizontal (convierte
la flecha 'recta y derecha' en 'recta e izquierda'), 80 épocas fijas y se evalúa last.pt;
val solo monitorea. Cada corrida es un proceso independiente y reanudable.

Uso: python train_r1.py --exp E2 --seed 0
     (exp en E1R, E2, E2noSV)
"""
import os, sys, json, argparse, gc

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
WEIGHTS = os.path.join(BASE, "best.pt")
PROJ = os.path.join(HERE, "runs", "r1")
DATA = {k: os.path.join(HERE, "dataset_v2", f"data_{k}.yaml") for k in ("E1R", "E2", "E2noSV", "E2noO")}
# sensibilidad: señales verticales de cruce enmascaradas en D1 (r1_senales_paso.py)
DATA.update({k: os.path.join(HERE, "dataset_v2c", f"data_{k}.yaml") for k in ("E1Rc", "E2c")})
EPOCHS = 80

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True, choices=list(DATA))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--smoke", action="store_true", help="1 época en carpeta aparte, para probar el pipeline")
a = ap.parse_args()
name = f"{a.exp}_s{a.seed}" + ("" if a.imgsz == 640 else f"_{a.imgsz}") + ("_smoke" if a.smoke else "")
if a.smoke:
    EPOCHS = 1
run_dir = os.path.join(PROJ, name)
done_flag = os.path.join(PROJ, "eval", f"{name}_test.json")
if os.path.exists(done_flag):
    print(f"{name}: ya terminado ({done_flag})")
    sys.exit(0)

import torch
from ultralytics import YOLO
sys.path.insert(0, HERE)
from r1_eval import evaluate

last = os.path.join(run_dir, "weights", "last.pt")
finished = False
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    finished = ck.get("epoch", -1) == -1  # Ultralytics marca epoch=-1 al terminar
if os.path.exists(last) and not finished:
    print(f"REANUDANDO {name} desde {last}", flush=True)
    YOLO(last).train(resume=True)
elif not finished:
    YOLO(WEIGHTS).train(
        data=DATA[a.exp], epochs=EPOCHS, patience=EPOCHS + 1, imgsz=a.imgsz,
        batch=2 if a.imgsz <= 640 else 1, device=0, workers=0, cache="disk",
        optimizer="AdamW", lr0=0.000625, lrf=0.01, momentum=0.9, weight_decay=0.0005, warmup_epochs=3,
        mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.0, flipud=0.0, degrees=0.0, translate=0.1, scale=0.5,
        seed=a.seed, deterministic=True, amp=True,
        project=PROJ, name=name, exist_ok=True, plots=True, verbose=False,
    )
gc.collect()
torch.cuda.empty_cache()
res = evaluate(last, DATA[a.exp], split="test", imgsz=a.imgsz, name=name)
print(f"{name} TERMINADO:", json.dumps(res["overall"]))
