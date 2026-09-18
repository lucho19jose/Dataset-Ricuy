# -*- coding: utf-8 -*-
"""Validación de la distancia monocular con medidas de cinta métrica desde la mototaxi (R1).

Etapas:
  python distance_validation.py calib  --video <tablero.mp4>             # f, c_x, c_y, distorsión (Zhang)
  python distance_validation.py theta  --video <marca8m.mp4> --row <px> # inclinación desde la marca a 8 m
  python distance_validation.py medir  --dir <carpeta_campo> --xlsx registro_campo.xlsx --weights <E2 last.pt>

Nada se ajusta con las tomas de prueba: f, c_y y la distorsión salen del tablero; theta sale de una marca
a 8 m que no forma parte de la prueba (o del inclinómetro, como control). Por cada toma se analizan los
fotogramas entre 1 s y 4 s y se toma la mediana; su dispersión mide la inestabilidad.

Estimadores:
  eq1_nominal : D = f*H/h_px con f = 950 px (QHAWAY)
  eq1_calib   : D = f*H/h_px con la f calibrada
  eq2         : D = h_cam / tan(theta + atan((y_b - c_y)/f))       (punto de contacto)
  media       : promedio simple de eq1_calib y eq2 (lo que hace la app, como QHAWAY), solo personas
  ponderada   : pesos 1/sigma^2 según la propagación de errores de cada ecuación, solo personas
"""
import os, sys, json, glob, argparse, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "r1", "distance")
CALIB = os.path.join(OUT, "calibracion.json")
H_REAL = {"Persona": 1.65, "Automóvil particular": 1.50, "Mototaxi": 1.60, "Motocicleta": 1.10}
F_NOMINAL = 950.0
SIG_H_REL, SIG_PX, SIG_THETA = 0.05, 2.0, math.radians(0.5)


def frames(video, t0=1.0, t1=4.0, step=3):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        t = i / fps
        if t0 <= t <= t1 and (i % step == 0):
            out.append(im)
        if t > t1:
            break
        i += 1
    cap.release()
    return out, fps


def calib(video, board=(9, 6), square_mm=25.0):
    cap = cv2.VideoCapture(video)
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_mm / 1000.0
    objs, imgs, size, i = [], [], None, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        i += 1
        if i % 10:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        found, c = cv2.findChessboardCorners(g, board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            objs.append(objp)
            imgs.append(c)
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    res = dict(n_vistas=len(objs), rms_px=float(rms), ancho=size[0], alto=size[1], fx=float(K[0, 0]),
               fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), dist=dist.ravel().tolist(),
               hfov_grados=float(2 * math.degrees(math.atan(size[0] / (2 * K[0, 0])))))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(CALIB, "w"), indent=1)
    print(json.dumps(res, indent=1))


def theta_from_mark(row_px, d_mark, h_cam):
    """Inclinación del eje óptico a partir de la fila de una marca en el suelo a distancia conocida."""
    c = json.load(open(CALIB))
    alpha = math.atan2(h_cam, d_mark)                       # ángulo real bajo la horizontal
    beta = math.atan2(row_px - c["cy"], c["fy"])            # ángulo de la fila respecto del eje óptico
    return alpha - beta


def estimates(h_px, y_b, cls, f, cy, h_cam, theta):
    H = H_REAL[cls]
    e = {"eq1_nominal": F_NOMINAL * H / h_px, "eq1_calib": f * H / h_px}
    a = theta + math.atan2(y_b - cy, f)
    e["eq2"] = h_cam / math.tan(a) if a > 0 else float("nan")
    if cls == "Persona" and not math.isnan(e["eq2"]):
        D = e["eq1_calib"]
        s1 = D * math.hypot(SIG_H_REL, SIG_PX * D / (f * H))
        s2 = (D ** 2 / h_cam) * math.hypot(SIG_THETA, SIG_PX / f)
        e["media"] = 0.5 * (e["eq1_calib"] + e["eq2"])
        w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
        e["ponderada"] = (w1 * e["eq1_calib"] + w2 * e["eq2"]) / (w1 + w2)
    return e


def medir(folder, xlsx, weights, h_cam, theta):
    import pandas as pd
    from ultralytics import YOLO
    c = json.load(open(CALIB))
    model = YOLO(weights)
    tomas = pd.read_excel(os.path.join(folder, xlsx), sheet_name="Tomas")
    rows = []
    for _, t in tomas.iterrows():
        if str(t["tipo"]) not in ("persona", "lateral", "vehículo") or pd.isna(t["archivo de video"]):
            continue
        cls = "Persona" if str(t["tipo"]) in ("persona", "lateral") else (
            "Mototaxi" if "Mototaxi" in str(t["objetivo"]) else "Automóvil particular")
        H_obj = float(t["altura real (m)"]) if not pd.isna(t["altura real (m)"]) else H_REAL[cls]
        ims, _ = frames(os.path.join(folder, str(t["archivo de video"])))
        per = []
        for im in ims:
            r = model.predict(im, imgsz=640, conf=0.25, verbose=False, device=0)[0]
            best = None
            for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                if r.names[int(k)] != cls:
                    continue
                xc = 0.5 * (b[0] + b[2])
                score = abs(xc - c["cx"])  # objetivo: la detección de la clase más cercana al eje
                if best is None or score < best[0]:
                    best = (score, b)
            if best is None:
                continue
            x0, y0, x1, y1 = best[1]
            pts = cv2.undistortPoints(np.array([[[0.5 * (x0 + x1), y0]], [[0.5 * (x0 + x1), y1]]], np.float32),
                                      np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]),
                                      np.array(c["dist"]), P=np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]))
            yt, yb = float(pts[0, 0, 1]), float(pts[1, 0, 1])
            e = estimates(yb - yt, yb, cls, c["fy"], c["cy"], h_cam, theta)
            e["h_px"], e["y_b"], e["recortado"] = yb - yt, yb, bool(y1 >= im.shape[0] - 2)
            if H_obj != H_REAL[cls]:  # control: la misma ecuación con la altura medida de la persona
                e["eq1_altura_medida"] = c["fy"] * H_obj / (yb - yt)
            per.append(e)
        if not per:
            rows.append(dict(toma=int(t["toma"]), clase=cls, D_real=float(t["distancia sobre el eje (m)"]), detectado=False))
            continue
        agg = {k: float(np.nanmedian([p[k] for p in per if k in p])) for k in per[0] if k != "recortado"}
        agg.update({f"{k}_de": float(np.nanstd([p[k] for p in per if k in p])) for k in ("eq1_calib", "eq2") if k in per[0]})
        agg.update(toma=int(t["toma"]), clase=cls, D_real=float(t["distancia sobre el eje (m)"]),
                   lateral=float(t["desplazamiento lateral (m)"] or 0), n_frames=len(per), detectado=True,
                   recortado=any(p["recortado"] for p in per))
        rows.append(agg)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "tomas_distancia.csv"), index=False, encoding="utf-8")
    methods = [m for m in ("eq1_nominal", "eq1_calib", "eq2", "media", "ponderada") if m in df]
    bands = [(0, 5.01), (5.01, 10.01), (10.01, 20.01)]
    summ = []
    for m in methods:
        for lo, hi in bands:
            s = df[(df.detectado) & (df.D_real > lo) & (df.D_real <= hi)].dropna(subset=[m])
            if len(s):
                err = s[m] - s.D_real
                summ.append(dict(metodo=m, tramo=f"{lo:.0f}-{hi:.0f} m", n=len(s), MAE=float(err.abs().mean()),
                                 sesgo=float(err.mean()), MAPE=float((err.abs() / s.D_real).mean() * 100)))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "resumen_distancia.csv"), index=False, encoding="utf-8")
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["calib", "theta", "medir"])
    ap.add_argument("--video")
    ap.add_argument("--row", type=float)
    ap.add_argument("--dmark", type=float, default=8.0)
    ap.add_argument("--hcam", type=float)
    ap.add_argument("--theta_deg", type=float)
    ap.add_argument("--dir")
    ap.add_argument("--xlsx", default="registro_campo.xlsx")
    ap.add_argument("--weights")
    a = ap.parse_args()
    if a.etapa == "calib":
        calib(a.video)
    elif a.etapa == "theta":
        th = theta_from_mark(a.row, a.dmark, a.hcam)
        print(f"theta = {math.degrees(th):.2f} grados (hacia abajo positivo)")
    else:
        medir(a.dir, a.xlsx, a.weights, a.hcam, math.radians(a.theta_deg))
