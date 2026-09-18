# -*- coding: utf-8 -*-
"""Latencia del modelo R1 sobre un video real (RTX 3050 portátil, 4 GB).

Mide, tras 30 fotogramas de calentamiento:
  - inferencia pura (preproceso, inferencia y posproceso por separado), FP32 y FP16;
  - la cadena con seguimiento ByteTrack (model.track), que aproxima la aplicación sin la interfaz ni la voz.
Uso: python r1_latency.py --weights runs/r1/E2_s0/weights/last.pt --video <video.mp4> [--n 600]
"""
import os, json, time, argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

frames = []
if a.video.endswith(".txt"):  # lista de imágenes (p. ej., fotogramas 1920x1080 del test), repetida hasta n
    paths = [p.strip() for p in open(a.video, encoding="utf-8") if p.strip()]
    paths = [p for p in paths if cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR).shape[1] == 1920]
    while len(frames) < a.n + 30:
        for p in paths:
            frames.append(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR))
            if len(frames) >= a.n + 30:
                break
else:
    cap = cv2.VideoCapture(a.video)
    while len(frames) < a.n + 30:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
res = {"video": os.path.basename(a.video), "resolucion": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
       "n_fotogramas": len(frames) - 30, "gpu": torch.cuda.get_device_name(0), "imgsz": a.imgsz}

for half in (False, True):
    model = YOLO(a.weights)
    sp = []
    for i, im in enumerate(frames):
        r = model.predict(im, imgsz=a.imgsz, half=half, conf=0.525, iou=0.45, verbose=False, device=0)[0]
        if i >= 30:
            sp.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    sp = np.array(sp)
    tot = sp.sum(1)
    res["FP16" if half else "FP32"] = dict(pre_ms=float(sp[:, 0].mean()), inferencia_ms=float(sp[:, 1].mean()),
                                         post_ms=float(sp[:, 2].mean()), total_ms=float(tot.mean()),
                                         p95_ms=float(np.percentile(tot, 95)), fps=float(1000 / tot.mean()))
    # cadena con seguimiento, medida de reloj de pared
    model = YOLO(a.weights)
    torch.cuda.synchronize()
    t0 = None
    for i, im in enumerate(frames):
        if i == 30:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        model.track(im, imgsz=a.imgsz, half=half, conf=0.525, iou=0.45, persist=True, tracker="bytetrack.yaml",
                    verbose=False, device=0)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / (len(frames) - 30)
    res["FP16" if half else "FP32"]["con_seguimiento_ms"] = wall * 1000
    res["FP16" if half else "FP32"]["con_seguimiento_fps"] = 1 / wall

out = os.path.join(HERE, "runs", "r1", f"latencia_{a.imgsz}.json")
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
