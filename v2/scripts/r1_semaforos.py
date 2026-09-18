# -*- coding: utf-8 -*-
"""Armonización de la convención de anotación de semáforos (R1, 18/09).

Problema: en parte del corpus el "semáforo" se anotó como el pórtico o el poste completo (con carteles y
brazo), y en otra parte como la cabeza del semáforo. Convención adoptada: un polígono por cabeza (la caja
con sus luces). El modelo base de QHAWAY detecta cabezas, así que se usa para PROPONER los reemplazos.

Para cada semáforo anotado:
  - si alguna cabeza detectada coincide con él (IoU >= 0.5), ya cumple la convención y se conserva;
  - si no, las cabezas detectadas (conf >= 0.15, clases 9 y 10, a 1024 px) cuyo centro cae dentro de su caja
    se proponen como reemplazo, con el color que eligió el anotador;
  - si no hay ninguna cabeza dentro, se marca para revisión manual.
Salidas: dataset_v2/semaforos_propuesta.json y una hoja de contactos para la revisión visual.
Uso: python r1_semaforos.py proponer | hoja
"""
import os, sys, json, glob
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "dataset_v2")
BASE_W = os.path.join(os.path.dirname(HERE), "best.pt")
PROP = os.path.join(D, "semaforos_propuesta.json")
SP = os.environ.get("SP", HERE)


def bbox(pts):
    xs, ys = zip(*pts)
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def key(uid, s):
    b = bbox(s["points"])
    return f"{uid}|{s['label']}|{round(b[0])},{round(b[1])},{round(b[2])},{round(b[3])}"


def proponer():
    from ultralytics import YOLO
    model = YOLO(BASE_W)
    out, stats = {}, {"conserva": 0, "reemplaza": 0, "sin_deteccion": 0}
    for jf in sorted(glob.glob(os.path.join(D, "unified_json", "*.json"))):
        uid = os.path.splitext(os.path.basename(jf))[0]
        d = json.load(open(jf, encoding="utf-8"))
        tl = [s for s in d["shapes"] if s["label"].startswith("Sem")]
        if not tl:
            continue
        img = os.path.join(D, "images", os.path.basename(d["imagePath"]))
        r = model.predict(img, imgsz=1024, conf=0.15, classes=[9, 10], verbose=False, device=0, retina_masks=True)[0]
        heads = []
        if r.masks is not None:
            for poly, b, c, cf in zip(r.masks.xy, r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if len(poly) >= 3:
                    step = max(1, len(poly) // 24)
                    heads.append((b, {"poly": [[float(x), float(y)] for x, y in poly[::step]], "cls": int(c), "conf": float(cf)}))
        for s in tl:
            gb = bbox(s["points"])
            if any(iou(gb, hb) >= 0.5 for hb, _ in heads):
                stats["conserva"] += 1
                continue
            pad_x, pad_y = 0.1 * (gb[2] - gb[0]) + 4, 0.1 * (gb[3] - gb[1]) + 4
            inside = [(hb, hp) for hb, hp in heads
                      if gb[0] - pad_x <= (hb[0] + hb[2]) / 2 <= gb[2] + pad_x
                      and gb[1] - pad_y <= (hb[1] + hb[3]) / 2 <= gb[3] + pad_y]
            k = key(uid, s)
            if inside:
                # color: el que predice el modelo si su confianza es alta (un pórtico puede tener cabezas en estados
                # distintos); si no, el que eligió el anotador
                nuevos = [dict(points=hp["poly"], label=("Semáforo en verde" if hp["cls"] == 9 else "Semáforo en rojo")
                               if hp["conf"] >= 0.5 else s["label"]) for _, hp in inside]
                out[k] = dict(uid=uid, label=s["label"], gt_box=gb, nuevos=nuevos,
                              nuevas_cajas=[hb for hb, _ in inside], estado="propuesto")
                stats["reemplaza"] += 1
            else:
                out[k] = dict(uid=uid, label=s["label"], gt_box=gb, nuevos=[], nuevas_cajas=[], estado="sin_deteccion")
                stats["sin_deteccion"] += 1
    json.dump(out, open(PROP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(stats, "->", PROP)


def hoja():
    prop = json.load(open(PROP, encoding="utf-8"))
    tiles = []
    for i, (k, p) in enumerate(prop.items()):
        d = json.load(open(os.path.join(D, "unified_json", p["uid"] + ".json"), encoding="utf-8"))
        im = Image.open(os.path.join(D, "images", os.path.basename(d["imagePath"]))).convert("RGB")
        gb = p["gt_box"]
        pad = max(30, 0.3 * max(gb[2] - gb[0], gb[3] - gb[1]))
        bx = (max(0, gb[0] - pad), max(0, gb[1] - pad), min(im.width, gb[2] + pad), min(im.height, gb[3] + pad))
        c = im.crop(bx).copy()
        dr = ImageDraw.Draw(c)
        dr.rectangle([gb[0] - bx[0], gb[1] - bx[1], gb[2] - bx[0], gb[3] - bx[1]], outline=(255, 0, 255), width=2)
        for hb in p["nuevas_cajas"]:
            dr.rectangle([hb[0] - bx[0], hb[1] - bx[1], hb[2] - bx[0], hb[3] - bx[1]], outline=(0, 255, 0), width=2)
        c.thumbnail((250, 170))
        tiles.append((f"{i}: {p['label'][12:]} {p['estado'][:4]} {len(p['nuevos'])}", c))
    cols, tw, th = 6, 255, 190
    for part in range(0, len(tiles), 60):
        T = tiles[part:part + 60]
        sheet = Image.new("RGB", (cols * tw, ((len(T) + cols - 1) // cols) * th), "white")
        dr = ImageDraw.Draw(sheet)
        for j, (cap, c) in enumerate(T):
            x, y = (j % cols) * tw, (j // cols) * th
            sheet.paste(c, (x + 3, y + 3))
            dr.text((x + 3, y + 175), cap, fill="black")
        sheet.save(os.path.join(SP, f"semaforos_{part // 60}.png"))
    print(len(tiles), "casos en", (len(tiles) + 59) // 60, "hojas")


if __name__ == "__main__":
    {"proponer": proponer, "hoja": hoja}[sys.argv[1]]()
