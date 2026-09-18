# -*- coding: utf-8 -*-
"""
Conversor LabelMe (poligonos) -> YOLO segmentacion, ALINEADO a best.pt.

A diferencia de labelme2yolo_seg.py, este usa EXACTAMENTE el orden y los
nombres de clase del modelo best.pt (entrenado en Ayacucho), para que el
fine-tuning conserve la cabeza de deteccion ya aprendida.

Mapea los sinonimos de tus etiquetas LabelMe a los nombres canonicos de best.pt:
    Tope de velocidad          -> Reductor de velocidad
    Cruce peatonal             -> Paso peatonal
    Autobus                    -> Bus de transporte
    Linea recta a la derecha   -> Linea recta y derecha
    (el resto se mapea por nombre directo)

Salida -> yoloe_autolabel/dataset_best/  (no toca dataset/)

Uso:
    python labelme2yolo_best.py
"""
import json, glob, os, shutil, random

BASE = r"C:\Users\José Luis\Desktop\1. MAESTRIA\CICLO II\DEEP LEARNING\Labelme"
SRC_DIR = os.path.join(BASE, "imagenes")
OUT_DIR = os.path.join(BASE, "yoloe_autolabel", "dataset_best")
VAL_RATIO = 0.20
SEED = 42
INCLUIR_COPIAS = False

# Orden EXACTO de clases en best.pt (NO cambiar):
CLASSES = [
    "Reductor de velocidad",   # 0
    "Paso peatonal",           # 1
    "Línea recta",             # 2
    "Línea recta y derecha",   # 3
    "Persona",                 # 4
    "Motocicleta",             # 5
    "Automóvil particular",    # 6
    "Camión",                  # 7
    "Bus de transporte",       # 8
    "Semáforo en verde",       # 9
    "Semáforo en rojo",        # 10
    "Mototaxi",                # 11
]
NAME2ID = {c: i for i, c in enumerate(CLASSES)}

# Sinonimos de LabelMe -> nombre canonico de best.pt
ALIAS = {
    "tope de velocidad": "Reductor de velocidad",
    "reductor de velocidad": "Reductor de velocidad",
    "cruce peatonal": "Paso peatonal",
    "paso peatonal": "Paso peatonal",
    "línea recta": "Línea recta",
    "linea recta": "Línea recta",
    "línea recta a la derecha": "Línea recta y derecha",
    "linea recta a la derecha": "Línea recta y derecha",
    "línea recta y derecha": "Línea recta y derecha",
    "persona": "Persona",
    "motocicleta": "Motocicleta",
    "automóvil particular": "Automóvil particular",
    "automovil particular": "Automóvil particular",
    "camión": "Camión",
    "camion": "Camión",
    "autobús": "Bus de transporte",
    "autobus": "Bus de transporte",
    "bus de transporte": "Bus de transporte",
    "semáforo en verde": "Semáforo en verde",
    "semaforo en verde": "Semáforo en verde",
    "semáforo en rojo": "Semáforo en rojo",
    "semaforo en rojo": "Semáforo en rojo",
    "mototaxi": "Mototaxi",
}


def label_a_id(label):
    canon = ALIAS.get((label or "").lower().strip())
    return NAME2ID[canon] if canon else None


def clamp01(v):
    return min(1.0, max(0.0, v))


def shape_a_poly_norm(shape, w, h):
    pts = shape.get("points", [])
    t = shape.get("shape_type")
    if t == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    elif t == "polygon":
        if len(pts) < 3:
            return None
    else:
        return None
    out = []
    for x, y in pts:
        out.append(clamp01(x / w))
        out.append(clamp01(y / h))
    return out


def main():
    random.seed(SEED)
    jsons = sorted(glob.glob(os.path.join(SRC_DIR, "*.json")))
    if not INCLUIR_COPIAS:
        jsons = [f for f in jsons if "- copia" not in os.path.basename(f)]

    # carpeta de salida limpia
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    items = list(jsons)
    random.shuffle(items)
    n_val = int(len(items) * VAL_RATIO)
    val_set = set(items[:n_val])

    stats = {"train": 0, "val": 0}
    inst_por_clase = {c: 0 for c in CLASSES}
    desconocidas = {}
    sin_imagen = []
    sin_shapes = 0

    for jf in items:
        d = json.load(open(jf, encoding="utf-8"))
        w, h = d.get("imageWidth"), d.get("imageHeight")
        img_name = os.path.basename(d.get("imagePath") or (os.path.splitext(os.path.basename(jf))[0] + ".jpg"))
        img_path = os.path.join(SRC_DIR, img_name)
        if not os.path.exists(img_path):
            alt = os.path.splitext(jf)[0] + ".jpg"
            if os.path.exists(alt):
                img_path, img_name = alt, os.path.basename(alt)
            else:
                sin_imagen.append(img_name)
                continue

        lines = []
        for s in d.get("shapes", []):
            cid = label_a_id(s.get("label", ""))
            if cid is None:
                lab = s.get("label")
                desconocidas[lab] = desconocidas.get(lab, 0) + 1
                continue
            poly = shape_a_poly_norm(s, w, h)
            if poly is None:
                continue
            lines.append(str(cid) + " " + " ".join(f"{v:.6f}" for v in poly))
            inst_por_clase[CLASSES[cid]] += 1

        if not lines:
            sin_shapes += 1
            continue

        split = "val" if jf in val_set else "train"
        shutil.copy2(img_path, os.path.join(OUT_DIR, "images", split, img_name))
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(OUT_DIR, "labels", split, txt_name), "w", encoding="utf-8") as fo:
            fo.write("\n".join(lines) + "\n")
        stats[split] += 1

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fo:
        fo.write(f"path: {OUT_DIR}\n")
        fo.write("train: images/train\n")
        fo.write("val: images/val\n\n")
        fo.write(f"nc: {len(CLASSES)}\n")
        fo.write("names:\n")
        for i, c in enumerate(CLASSES):
            fo.write(f"  {i}: {c}\n")

    print("=" * 60)
    print("CONVERSION LABELME -> YOLO-SEG (alineado a best.pt) COMPLETADA")
    print("=" * 60)
    print(f"JSON procesados:      {len(items)}")
    print(f"Imagenes convertidas: {stats['train']+stats['val']}  (train={stats['train']}, val={stats['val']})")
    print(f"JSON sin formas:      {sin_shapes}")
    if sin_imagen:
        print(f"JSON sin imagen:      {len(sin_imagen)}  ej: {sin_imagen[:3]}")
    print("\nInstancias por clase:")
    for c in CLASSES:
        print(f"  {NAME2ID[c]:2d}  {c:24s} {inst_por_clase[c]}")
    if desconocidas:
        print("\n[!] Etiquetas DESCONOCIDAS (revisar):")
        for k, v in desconocidas.items():
            print(f"  {v:5d}  {k!r}")
    print(f"\ndata.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
