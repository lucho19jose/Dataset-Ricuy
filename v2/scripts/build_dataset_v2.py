# -*- coding: utf-8 -*-
"""Dataset v2 (revisión R1 del artículo RICUY).

Unifica imagenes/ (253 imágenes del experimento 1) con las anotaciones nuevas de
Mario, Oscar y Paul, deduplica, fusiona fragmentos, agrupa en el tiempo y parte en
train/val/test POR GRUPOS (sin fuga entre frames vecinos). No modifica ningún
archivo original: todo se escribe en yoloe_autolabel/dataset_v2/.

Etapas (en orden):
    python build_dataset_v2.py collect     # unifica y deduplica -> dataset_v2/images + unified_json
    python build_dataset_v2.py qa          # hoja de contactos de los semáforos de Paul
    python build_dataset_v2.py candidates  # candidatos de marcas viales en imágenes antiguas (E0, 1024 px)
    python build_dataset_v2.py merge       # incorpora los JSON revisados en LabelMe (flag 'revisado')
    python build_dataset_v2.py split       # grupos + split 70/10/20 + labels YOLO + yaml + manifiesto

Correcciones manuales de semáforos: dataset_v2/qa_semaforos.csv (uid,shape_idx,accion)
con accion en {keep, rojo, verde, mask}. 'collect' las aplica si el archivo existe.
"""
import os, re, sys, json, glob, shutil, hashlib, random, unicodedata, csv, collections
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelme2yolo_best import CLASSES, NAME2ID, ALIAS  # mismo esquema y sinónimos del E1

BASE = r"C:\Users\José Luis\Desktop\1. MAESTRIA\CICLO II\DEEP LEARNING\Labelme"
OLD_DIR = os.path.join(BASE, "imagenes")
NEW_DIR = os.path.join(BASE, "OBSERVACIONES", "new dataset labeled")
OUT = os.path.join(BASE, "yoloe_autolabel", "dataset_v2")
IMG_OUT = os.path.join(OUT, "images")
JSON_OUT = os.path.join(OUT, "unified_json")
LBL_OUT = os.path.join(OUT, "labels")
REVIEW_DIR = os.path.join(OUT, "revisar_marcas_viales")
BASE_WEIGHTS = os.path.join(BASE, "best.pt")  # modelo QHAWAY (Ayacucho), entrenado a 1024 px

# alias nuevos detectados en la auditoría de los datos de Paul
ALIAS = dict(ALIAS)
ALIAS.update({"reductor vehicular": "Reductor de velocidad", "pase peatonal": "Paso peatonal"})

RARE = [0, 1, 2, 3]            # marcas viales / clases escasas
GAP_S = 30.0                   # corte temporal: >30 s entre frames del mismo video (calibrado con pHash, 18/09)
PHASH_MAX = 10                 # casi duplicado: frames a >60 s nunca bajan de 16 (p10 = 24)
MIN_AREA = 20.0                # px²: fragmentos huérfanos menores se descartan
SPLIT_SEED = 2026
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}

T_MARIO = re.compile(r"t(\d+)m(\d+)-(\d+)s")
T_LIMA = re.compile(r"^(?P<vid>.+?)_t(?P<ms>\d+)ms_f\d+$")
T_ANDA = re.compile(r"^video_Andahuaylas(?:_(?P<v>\d))?_(?P<i>\d{4})$")
T_OSCAR = re.compile(r"^(?P<vid>VID_\d+_\d+)_(?P<i>\d{4})$")


# ------------------------------------------------------------------ utilidades
def canon(label):
    c = ALIAS.get((label or "").lower().strip())
    return c


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def phash(path):
    """pHash de 64 bits (DCT 32x32 -> 8x8) sin dependencias extra."""
    import cv2
    g = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = cv2.dct(g)[:8, :8]
    med = np.median(d.ravel()[1:])
    return (d > med).ravel()


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return re.sub(r"_+", "_", s)


def poly_area(pts):
    if len(pts) < 3:
        return 0.0
    x, y = np.asarray(pts, float).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def shape_points(s):
    """Devuelve los puntos de la forma como polígono, o None si no es usable."""
    pts, t = s.get("points", []), s.get("shape_type")
    if t == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [[min(x1, x2), min(y1, y2)], [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)], [min(x1, x2), max(y1, y2)]]
    if t in ("polygon", "linestrip") and len(pts) >= 3:
        return [list(map(float, p)) for p in pts]
    return None


# ------------------------------------------------------------------ recolección
def source_meta(json_path):
    """Fuente, video, tiempo (s) y ubicación de cada imagen, según su carpeta y nombre."""
    stem = os.path.splitext(os.path.basename(json_path))[0]
    if os.path.dirname(json_path) == OLD_DIR:
        m = T_ANDA.match(stem)
        if m:
            v = m.group("v") or "1"
            return dict(src="A", video=f"Andahuaylas_{v}", t=float(m.group("i")), loc="")
        m = T_LIMA.match(stem)
        if m:
            return dict(src="L", video=m.group("vid"), t=int(m.group("ms")) / 1000.0, loc="")
        raise ValueError(f"nombre antiguo no reconocido: {stem}")
    who = os.path.relpath(json_path, NEW_DIR).split(os.sep)[0]
    if who == "OSCAR":
        m = T_OSCAR.match(stem)
        return dict(src="O", video=m.group("vid"), t=float(m.group("i")), loc="")  # ~1 frame/s
    m = T_MARIO.search(stem)
    if who in ("MARIO", "PAUL") and m:
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
        return dict(src="M" if who == "MARIO" else "P", video="MARIO_47min", t=t, loc="")
    if who == "PAUL":  # captura de Google Street View, agrupada por distrito
        name = re.sub(r"\s+", " ", stem.replace("#U00f1", "ñ")).strip()
        loc = re.sub(r"\s+\d+.*$", "", name).strip() or name
        return dict(src="SV", video="", t=0.0, loc=slug(loc).lower())
    raise ValueError(f"fuente no reconocida: {json_path}")


def priority(rec):
    """Para duplicados: Mario > Oscar > Paul flechas_ok_3 > resto de Paul."""
    p = rec["json"]
    if os.sep + "MARIO" + os.sep in p:
        return 0
    if os.sep + "OSCAR" + os.sep in p:
        return 1
    if "flechas_ok_3" in p:
        return 2
    return 3


def load_qa():
    path = os.path.join(OUT, "qa_semaforos.csv")
    qa = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qa[(row["uid"], int(row["shape_idx"]))] = row["accion"].strip().lower()
    return qa


def collect():
    old = [f for f in sorted(glob.glob(os.path.join(OLD_DIR, "*.json"))) if "- copia" not in f]
    new = sorted(glob.glob(os.path.join(NEW_DIR, "**", "*.json"), recursive=True))
    recs, log = [], collections.Counter()
    for jf in old + new:
        d = json.load(open(jf, encoding="utf-8"))
        img = os.path.join(os.path.dirname(jf), os.path.basename(d["imagePath"]))
        if not os.path.exists(img):
            log["json_sin_imagen"] += 1
            continue
        meta = source_meta(jf)
        recs.append(dict(json=jf, img=img, data=d, md5=md5(img), **meta))

    # deduplicación exacta (MD5) con prioridad de anotador
    recs.sort(key=priority)
    seen, uniq = {}, []
    for r in recs:
        if r["md5"] in seen:
            log["duplicado_md5"] += 1
            continue
        seen[r["md5"]] = r
        uniq.append(r)

    qa = load_qa()
    hp = os.path.join(OUT, "semaforos_propuesta.json")
    harmon = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else {}
    cp = os.path.join(OUT, "candidatos_revision_test.json")
    cand_rev = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else {}
    fp = os.path.join(OUT, "correcciones_test.json")
    fixes = {f["clave"]: f for f in json.load(open(fp, encoding="utf-8"))} if os.path.exists(fp) else {}
    for d in (IMG_OUT, JSON_OUT):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    changelog = []
    used = set()
    manifest = []
    for r in uniq:
        stem = os.path.splitext(os.path.basename(r["img"]))[0]
        uid = f"{r['src']}_{slug(stem)}"
        while uid in used:
            uid += "_b"
        used.add(uid)
        im = Image.open(r["img"])
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        ext = ".png" if r["src"] == "SV" else ".jpg"

        # formas: normalizar etiqueta, convertir, fusionar fragmentos con group_id
        groups, shapes = collections.defaultdict(list), []
        for s in r["data"]["shapes"]:
            lab = canon(s.get("label"))
            if lab is None:
                log[f"etiqueta_desconocida:{s.get('label')}"] += 1
                continue
            pts = shape_points(s)
            if pts is None:
                log[f"forma_descartada:{s.get('shape_type')}"] += 1
                continue
            if s.get("shape_type") == "linestrip":
                log["linestrip_a_poligono"] += 1
            gid = s.get("group_id")
            if gid is not None:
                groups[(lab, gid)].append(pts)
            else:
                shapes.append([lab, pts])
        from ultralytics.data.converter import merge_multi_segment
        for (lab, gid), segs in groups.items():
            if len(segs) == 1:
                shapes.append([lab, segs[0]])
            else:
                merged = np.concatenate(merge_multi_segment([np.asarray(s).ravel().tolist() for s in segs]), 0)
                shapes.append([lab, merged.tolist()])
                log["fragmentos_fusionados"] += len(segs) - 1
        kept = []
        for lab, pts in shapes:
            if poly_area(pts) < MIN_AREA:
                log["poligono_diminuto_descartado"] += 1
                continue
            kept.append([lab, pts])

        # correcciones manuales de semáforos (QA visual)
        final, draw = [], None
        for i, (lab, pts) in enumerate(kept):
            act = qa.get((uid, i))
            if act in ("rojo", "verde"):
                new_lab = "Semáforo en rojo" if act == "rojo" else "Semáforo en verde"
                if new_lab != lab:
                    changelog.append([uid, i, lab, new_lab, "color corregido en QA visual"])
                lab = new_lab
            elif act == "mask":
                draw = draw or ImageDraw.Draw(im)
                xs, ys = zip(*pts)
                draw.rectangle([min(xs), min(ys), max(xs), max(ys)], fill=(114, 114, 114))
                changelog.append([uid, i, lab, "(enmascarado)", "caja de pórtico sin arreglo: región en gris"])
                continue
            final.append(dict(label=lab, points=pts, group_id=None, shape_type="polygon",
                              flags={}, description=""))
        # armonización de semáforos: un polígono por cabeza (ver r1_semaforos.py y su revisión visual)
        if harmon:
            kept_final = []
            for s in final:
                xs, ys = zip(*s["points"])
                k = f"{uid}|{s['label']}|{round(min(xs))},{round(min(ys))},{round(max(xs))},{round(max(ys))}"
                p = harmon.get(k)
                if p is None:
                    kept_final.append(s)
                elif p["estado"] == "propuesto" and p["nuevos"]:
                    for nv in p["nuevos"]:
                        kept_final.append(dict(label=nv["label"], points=nv["points"], group_id=None,
                                               shape_type="polygon", flags={}, description=""))
                    changelog.append([uid, -1, s["label"], f"{len(p['nuevos'])} cabezas",
                                      "semáforo anotado como pórtico: reemplazado por cabezas"])
                else:
                    draw = draw or ImageDraw.Draw(im)
                    draw.rectangle([min(xs), min(ys), max(xs), max(ys)], fill=(114, 114, 114))
                    changelog.append([uid, -1, s["label"], "(enmascarado)",
                                      "semáforo sin cabeza identificable: región en gris"])
            final = kept_final
        # candidatos del modelo base revisados visualmente (solo imágenes de prueba; ver candidatos_revision_test.json)
        for c in cand_rev.get(uid, []):
            if c["decision"] == "acepta":
                final.append(dict(label=c["label"], points=c["points"], group_id=None, shape_type="polygon",
                                  flags={}, description=""))
                changelog.append([uid, -1, "(sin anotar)", c["label"], "candidato aceptado: " + c["motivo"]])
        # correcciones de errores de anotación detectados en la revisión (ver correcciones_test.json)
        if fixes:
            corr = []
            for s in final:
                xs, ys = zip(*s["points"])
                k = f"{uid}|{s['label']}|{round(min(xs))},{round(min(ys))},{round(max(xs))},{round(max(ys))}"
                fx = fixes.get(k)
                if fx is None:
                    corr.append(s)
                elif fx["accion"] == "reetiquetar":
                    changelog.append([uid, -1, s["label"], fx["nuevo"], fx["motivo"]])
                    s = dict(s, label=fx["nuevo"])
                    corr.append(s)
                else:
                    changelog.append([uid, -1, s["label"], "(borrado)", fx["motivo"]])
            final = corr
        im.save(os.path.join(IMG_OUT, uid + ext), quality=95) if ext == ".jpg" else im.save(os.path.join(IMG_OUT, uid + ext))
        out = dict(version="5.5.0", flags={}, shapes=final, imagePath=f"../images/{uid}{ext}",
                   imageData=None, imageHeight=h, imageWidth=w)
        json.dump(out, open(os.path.join(JSON_OUT, uid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        manifest.append(dict(uid=uid, file=uid + ext, src=r["src"], video=r["video"], t=r["t"], loc=r["loc"],
                             md5=r["md5"], origen=os.path.relpath(r["json"], BASE), w=w, h=h,
                             n_obj=len(final)))
    with open(os.path.join(OUT, "collect_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "changelog_etiquetas.csv"), "w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["uid", "shape_idx", "antes", "despues", "motivo"])
        wcsv.writerows(changelog)
    cnt = collections.Counter(m["src"] for m in manifest)
    print("imágenes únicas por fuente:", dict(cnt), "| total", len(manifest))
    for k, v in sorted(log.items()):
        print(f"  {k}: {v}")


# ------------------------------------------------------------------ QA semáforos
def qa_sheet():
    man = json.load(open(os.path.join(OUT, "collect_manifest.json"), encoding="utf-8"))
    tiles = []
    for m in man:
        if m["src"] not in ("P", "SV"):
            continue
        d = json.load(open(os.path.join(JSON_OUT, m["uid"] + ".json"), encoding="utf-8"))
        im = Image.open(os.path.join(IMG_OUT, m["file"]))
        for i, s in enumerate(d["shapes"]):
            if not s["label"].startswith("Sem"):
                continue
            xs, ys = zip(*s["points"])
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            pad = max(20, 0.6 * max(x1 - x0, y1 - y0))
            crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
            crop = crop.copy()
            ImageDraw.Draw(crop).rectangle([x0 - max(0, x0 - pad), y0 - max(0, y0 - pad),
                                            x1 - max(0, x0 - pad), y1 - max(0, y0 - pad)], outline=(0, 255, 255), width=2)
            crop.thumbnail((300, 300))
            tiles.append((f"{m['uid']}#{i} {s['label'][12:]}", crop))
    cols, tw, th = 4, 320, 340
    sheet = Image.new("RGB", (cols * tw, ((len(tiles) + cols - 1) // cols) * th), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (cap, crop) in enumerate(tiles):
        x, y = (k % cols) * tw, (k // cols) * th
        sheet.paste(crop, (x + 10, y + 10))
        dr.text((x + 10, y + 315), cap[:48], fill="black")
    path = os.path.join(OUT, "qa_semaforos_paul.png")
    sheet.save(path)
    print(len(tiles), "semáforos ->", path)


# ------------------------------------------------------------------ candidatos
def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


CAND_CLASSES = [0, 1]
# Solo reductor y paso peatonal. Revisión visual (18/09): con las clases 2 y 3 el modelo
# base de QHAWAY marca los segmentos discontinuos de carril, no las flechas pintadas que
# nuestro esquema llama "Línea recta" / "Línea recta y derecha". Sus candidatos serían
# ruido; las flechas faltantes se buscan después con el modelo E2.


def candidates(conf=0.35, weights=None, classes=None, only_split=None, review_dir=None, imgsz=1024):
    """Candidatos de marcas viales sin etiquetar en las imágenes antiguas (fuentes L y A).
    Por defecto: modelo base a 1024 px, clases 0 y 1. Con --weights/--classes/--split se reutiliza con el
    modelo E2 para buscar flechas (clases 2 y 3) solo en test."""
    from ultralytics import YOLO
    global REVIEW_DIR
    REVIEW_DIR = review_dir or REVIEW_DIR
    classes = classes if classes is not None else CAND_CLASSES
    man = json.load(open(os.path.join(OUT, "collect_manifest.json"), encoding="utf-8"))
    if only_split:
        fr = json.load(open(os.path.join(OUT, "split_frozen.json"), encoding="utf-8"))["asignacion"]
        man = [m for m in man if fr.get(m["uid"]) == only_split]
    model = YOLO(weights or BASE_WEIGHTS)
    if os.path.isdir(REVIEW_DIR):
        done = [j for j in glob.glob(os.path.join(REVIEW_DIR, "*.json"))
                if json.load(open(j, encoding="utf-8")).get("flags", {}).get("revisado")]
        if done:  # nunca borrar trabajo humano ya revisado
            sys.exit(f"{len(done)} JSON ya marcados como revisados en {REVIEW_DIR}; no se regeneran.")
        shutil.rmtree(REVIEW_DIR)
    os.makedirs(REVIEW_DIR)
    n_img, per_cls = 0, collections.Counter()
    for m in man:
        if m["src"] not in ("L", "A"):
            continue
        img_path = os.path.join(IMG_OUT, m["file"])
        d = json.load(open(os.path.join(JSON_OUT, m["uid"] + ".json"), encoding="utf-8"))
        gt = []
        for s in d["shapes"]:
            if NAME2ID[s["label"]] in RARE:
                xs, ys = zip(*s["points"])
                gt.append((min(xs), min(ys), max(xs), max(ys)))
        r = model.predict(img_path, imgsz=imgsz, conf=conf, classes=classes, verbose=False, device=0,
                          retina_masks=True)[0]
        if r.masks is None:
            continue
        new_shapes = []
        for poly, box, c, p in zip(r.masks.xy, r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if len(poly) < 3 or any(box_iou(box, g) >= 0.3 for g in gt):
                continue
            step = max(1, len(poly) // 40)  # polígono liviano para editar en LabelMe
            pts = [[float(x), float(y)] for x, y in poly[::step]]
            new_shapes.append(dict(label=CLASSES[int(c)], points=pts, group_id=None, shape_type="polygon",
                                   flags={"candidato": True}, description=f"CANDIDATO modelo conf={p:.2f}"))
            per_cls[CLASSES[int(c)]] += 1
        if new_shapes:
            shutil.copy2(img_path, os.path.join(REVIEW_DIR, m["file"]))
            d["shapes"] = d["shapes"] + new_shapes
            d["imagePath"] = m["file"]
            d["flags"] = {"revisado": False}
            json.dump(d, open(os.path.join(REVIEW_DIR, m["uid"] + ".json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            n_img += 1
    print(f"{n_img} imágenes con candidatos -> {REVIEW_DIR}")
    for k, v in per_cls.most_common():
        print(f"  {k}: {v}")


def merge_reviewed():
    """Reemplaza el JSON unificado por el revisado, solo si el revisor marcó 'revisado'."""
    n_ok, n_pend = 0, 0
    added = collections.Counter()
    for jf in sorted(glob.glob(os.path.join(REVIEW_DIR, "*.json"))):
        d = json.load(open(jf, encoding="utf-8"))
        uid = os.path.splitext(os.path.basename(jf))[0]
        if not d.get("flags", {}).get("revisado"):
            n_pend += 1
            continue
        target = os.path.join(JSON_OUT, uid + ".json")
        base = json.load(open(target, encoding="utf-8"))
        shapes = []
        for s in d["shapes"]:
            if s.get("flags", {}).get("candidato"):
                added[s["label"]] += 1
            s["flags"], s["description"] = {}, ""
            shapes.append(s)
        base["shapes"] = shapes
        json.dump(base, open(target, "w", encoding="utf-8"), ensure_ascii=False)
        n_ok += 1
    print(f"revisados incorporados: {n_ok} | pendientes (sin marcar 'revisado'): {n_pend}")
    print("candidatos aceptados por clase:", dict(added))


# ------------------------------------------------------------------ split
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def u(self, a, b):
        self.p[self.f(a)] = self.f(b)


def build_groups(man):
    n = len(man)
    uf = UF(n)
    by_video = collections.defaultdict(list)
    for i, m in enumerate(man):
        if m["src"] == "A":
            by_video[("A", m["video"])].append(i)       # Andahuaylas: video completo = un grupo
        elif m["src"] == "SV":
            by_video[("SV", m["loc"])].append(i)         # Street View: por distrito
        else:
            by_video[("V", m["video"])].append(i)
    for (kind, _), idx in by_video.items():
        if kind in ("A", "SV"):
            for j in idx[1:]:
                uf.u(idx[0], j)
            continue
        idx.sort(key=lambda i: man[i]["t"])
        for a, b in zip(idx, idx[1:]):
            if man[b]["t"] - man[a]["t"] <= GAP_S:
                uf.u(a, b)
    # casi duplicados visuales (semáforo en rojo, frames recodificados)
    hashes = [phash(os.path.join(IMG_OUT, m["file"])) for m in man]
    H = np.stack(hashes)
    n_ph = 0
    for i in range(n):
        dist = np.count_nonzero(H[i + 1:] != H[i], axis=1)
        for j in np.nonzero(dist <= PHASH_MAX)[0]:
            if uf.f(i) != uf.f(i + 1 + j):
                n_ph += 1
            uf.u(i, i + 1 + j)
    gid = {}
    return [gid.setdefault(uf.f(i), len(gid)) for i in range(n)], n_ph


def class_counts(uid):
    d = json.load(open(os.path.join(JSON_OUT, uid + ".json"), encoding="utf-8"))
    c = np.zeros(len(CLASSES), int)
    for s in d["shapes"]:
        c[NAME2ID[s["label"]]] += 1
    return c


def assign(groups, gcount, gimgs, gsrc, n_iter=20000):
    """Busca la asignación de grupos que mejor respeta 70/10/20 por clase, con restricciones."""
    rng = random.Random(SPLIT_SEED)
    G = list(groups)
    total_c = sum(gcount[g] for g in G)
    total_i = sum(gimgs[g] for g in G)
    anda = sorted([g for g in G if gsrc[g] == "A"], key=lambda g: -gimgs[g])
    best, best_score = None, 1e18
    for _ in range(n_iter):
        order = [g for g in G if gsrc[g] != "A"]
        rng.shuffle(order)
        sp = {}
        a = anda[:]
        rng.shuffle(a)
        for k, g in enumerate(a):                       # 2 videos de Andahuaylas a test
            sp[g] = "test" if k < 2 else "train"
        acc = {s: 0 for s in RATIOS}
        for g in anda:
            acc[sp[g]] += gimgs[g]
        for g in order:
            need = {s: RATIOS[s] * total_i - acc[s] for s in RATIOS}
            s = max(need, key=need.get)
            sp[g] = s
            acc[s] += gimgs[g]
        score = 0.0
        for s, r in RATIOS.items():
            cs = sum(gcount[g] for g in G if sp[g] == s)
            share = cs / np.maximum(total_c, 1)
            w = np.where(np.isin(np.arange(len(CLASSES)), RARE), 4.0, 1.0)
            score += float(np.sum(w * (share - r) ** 2))
            score += 2.0 * (acc[s] / total_i - r) ** 2
        for c in RARE:                                   # ≥3 grupos de cada clase escasa en test, ≥1 en val
            gt = sum(1 for g in G if sp[g] == "test" and gcount[g][c] > 0)
            gv = sum(1 for g in G if sp[g] == "val" and gcount[g][c] > 0)
            score += 1.0 * max(0, 3 - gt) + 0.5 * max(0, 1 - gv)
        old_val = sum(gimgs[g] for g in G if sp[g] == "val" and gsrc[g] in ("L", "A"))
        score += 0.5 * (old_val < 15)                    # E1-R necesita un val mínimo
        if score < best_score:
            best, best_score = dict(sp), score
    return best, best_score


def split():
    man = json.load(open(os.path.join(OUT, "collect_manifest.json"), encoding="utf-8"))
    gids, n_ph = build_groups(man)
    counts = {m["uid"]: class_counts(m["uid"]) for m in man}
    gcount = collections.defaultdict(lambda: np.zeros(len(CLASSES), int))
    gimgs, gsrc = collections.Counter(), {}
    for m, g in zip(man, gids):
        gcount[g] += counts[m["uid"]]
        gimgs[g] += 1
        gsrc[g] = "SV" if m["src"] == "SV" else ("A" if m["src"] == "A" else gsrc.get(g, m["src"]))
    frozen_path = os.path.join(OUT, "split_frozen.json")
    if os.path.exists(frozen_path) and "--refreeze" not in sys.argv:
        # el split ya está congelado: solo se reexportan etiquetas (p. ej., tras incorporar revisiones)
        frozen = json.load(open(frozen_path, encoding="utf-8"))
        sp_uid, score = frozen["asignacion"], frozen["puntaje"]
        print("split congelado reutilizado:", frozen_path)
    else:
        sp, score = assign(set(gids), gcount, gimgs, gsrc)
        sp_uid = {m["uid"]: sp[g] for m, g in zip(man, gids)}
        json.dump(dict(asignacion=sp_uid, puntaje=score, semilla=SPLIT_SEED, gap_s=GAP_S, phash_max=PHASH_MAX),
                  open(frozen_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)

    # etiquetas YOLO (polígonos normalizados)
    if os.path.isdir(LBL_OUT):
        shutil.rmtree(LBL_OUT)
    os.makedirs(LBL_OUT)
    for m in man:
        d = json.load(open(os.path.join(JSON_OUT, m["uid"] + ".json"), encoding="utf-8"))
        lines = []
        for s in d["shapes"]:
            xy = " ".join(f"{min(1, max(0, x / m['w'])):.6f} {min(1, max(0, y / m['h'])):.6f}" for x, y in s["points"])
            lines.append(f"{NAME2ID[s['label']]} {xy}")
        with open(os.path.join(LBL_OUT, m["uid"] + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # listas por experimento
    sdir = os.path.join(OUT, "splits")
    os.makedirs(sdir, exist_ok=True)
    rows = []
    lists = collections.defaultdict(list)
    for m, g in zip(man, gids):
        s = sp_uid[m["uid"]]
        p = os.path.join(IMG_OUT, m["file"])
        rows.append(dict(uid=m["uid"], archivo=m["file"], fuente=m["src"], video=m["video"], t=m["t"],
                         ubicacion=m["loc"], grupo=g, split=s, md5=m["md5"], origen=m["origen"],
                         **{f"c{k}": int(v) for k, v in enumerate(counts[m["uid"]])}))
        old = m["src"] in ("L", "A")
        if s == "test":
            lists["test"].append(p)
            lists["test_sinSV" if m["src"] != "SV" else "test_SV"].append(p)
            lists[f"test_src_{m['src']}"].append(p)
        else:
            lists[f"E2_{s}"].append(p)
            if m["src"] != "SV":
                lists[f"E2noSV_{s}"].append(p)
            if old:
                lists[f"E1R_{s}"].append(p)
            if m["src"] != "O":  # control: E2 sin el recorrido del 12/09 (se evalúa solo en ese recorrido)
                lists[f"E2noO_{s}"].append(p)
    for k, v in lists.items():
        with open(os.path.join(sdir, k + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(v)) + "\n")

    def yaml(name, train, val, test):
        y = [f"path: '{OUT}'", f"train: splits/{train}.txt", f"val: splits/{val}.txt",
             f"test: splits/{test}.txt", f"nc: {len(CLASSES)}", "names:"]
        y += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
        with open(os.path.join(OUT, f"data_{name}.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join(y) + "\n")
    yaml("E1R", "E1R_train", "E1R_val", "test")
    yaml("E2", "E2_train", "E2_val", "test")
    yaml("E2noSV", "E2noSV_train", "E2noSV_val", "test")
    yaml("E2noO", "E2noO_train", "E2noO_val", "test")
    for sub in [k for k in lists if k.startswith("test_")]:
        yaml(sub, "E2_train", sub, sub)  # solo para evaluar subconjuntos del test

    with open(os.path.join(OUT, "split_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    report(rows, gids, n_ph, score)


def report(rows, gids, n_ph, score):
    L = []
    L.append("# Dataset v2 (R1): resumen\n")
    L.append(f"Imágenes: {len(rows)} | grupos: {len(set(gids))} | uniones por pHash: {n_ph} | "
             f"puntaje del split: {score:.4f} | semilla {SPLIT_SEED} | corte temporal {GAP_S:.0f} s\n")
    L.append("## Imágenes y grupos por fuente y split\n")
    L.append("| Fuente | train | val | test | grupos |\n|---|--:|--:|--:|--:|")
    names = {"L": "Lima (E1)", "A": "Andahuaylas (E1)", "M": "Mario (video)", "O": "Oscar (video)",
             "P": "Paul (frames del video de Mario)", "SV": "Paul (Street View)"}
    for src in ["L", "A", "M", "O", "P", "SV"]:
        r = [x for x in rows if x["fuente"] == src]
        if not r:
            continue
        c = collections.Counter(x["split"] for x in r)
        L.append(f"| {names[src]} | {c['train']} | {c['val']} | {c['test']} | {len(set(x['grupo'] for x in r))} |")
    L.append("\n## Instancias por clase y split (entre paréntesis: grupos con la clase)\n")
    L.append("| ID | Clase | train | val | test | total |\n|--:|---|--:|--:|--:|--:|")
    for k, c in enumerate(CLASSES):
        cells = []
        tot = 0
        for s in ("train", "val", "test"):
            inst = sum(x[f"c{k}"] for x in rows if x["split"] == s)
            ng = len({x["grupo"] for x in rows if x["split"] == s and x[f"c{k}"] > 0})
            cells.append(f"{inst} ({ng})")
            tot += inst
        L.append(f"| {k} | {c} | {' | '.join(cells)} | {tot} |")
    tot_c = np.array([sum(x[f"c{k}"] for x in rows) for k in range(len(CLASSES))])
    mx = tot_c.max()
    L.append("\n## Desbalance (clase mayoritaria / clase)\n")
    for k in np.argsort(tot_c):
        L.append(f"- {CLASSES[k]}: {tot_c[k]} instancias, razón {mx / max(tot_c[k], 1):.1f}:1")
    with open(os.path.join(OUT, "resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(OUT, exist_ok=True)
    if step == "candidates" and "--weights" in sys.argv:
        # ej.: candidates --weights runs/r1/E2_s0/weights/last.pt --classes 2,3 --split test --out revisar_flechas_test
        arg = lambda k: sys.argv[sys.argv.index(k) + 1]
        candidates(conf=0.4, weights=arg("--weights"), classes=[int(c) for c in arg("--classes").split(",")],
                   only_split=arg("--split"), review_dir=os.path.join(OUT, arg("--out")), imgsz=640)
    else:
        {"collect": collect, "qa": qa_sheet, "candidates": candidates, "merge": merge_reviewed,
         "split": split}.get(step, lambda: print(__doc__))()
