# -*- coding: utf-8 -*-
"""Figuras del artículo R1, dibujadas al tamaño final de impresión.

Todas las cifras se leen de los archivos del experimento (split_manifest.csv, runs/r1/eval/*.json);
nada se copia a mano. Una columna = 8.5 cm, ancho completo = 18 cm; texto de 8 a 9 pt en Times.
Paleta categórica validada (skill dataviz): E2 azul, E1-R naranja, E0 aguamarina, E2 sin SV amarillo,
siempre con marcador distinto por serie.

Uso: python make_figures_r1.py [arquitectura|pinhole|montaje|distribucion|todas]
"""
import os, sys, csv, json, collections, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "RICUY_ADAS_UNI_2026_R1", "Figures", "r1")
os.makedirs(OUT, exist_ok=True)
CM = 1 / 2.54
COL, FULL = 8.5 * CM, 18.0 * CM

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
SERIES = {"E2": ("#2a78d6", "o"), "E1R": ("#eb6834", "s"), "E0": ("#1baf7a", "^"), "E2noSV": ("#eda100", "D")}
SHORT = ["Reductor", "Paso peatonal", "Flecha recta", "Flecha recta y der.", "Persona", "Motocicleta",
         "Automóvil", "Camión", "Bus", "Semáforo verde", "Semáforo rojo", "Mototaxi"]
MANIFEST = os.path.join(HERE, "dataset_v2", "split_manifest.csv")


def save(fig, name, engine=True):
    """Guarda con el tamaño EXACTO de la figura (sin recorte 'tight'), para que LaTeX la incluya a escala 1
    y el texto conserve su tamaño en puntos. El layout restringido acomoda leyendas y rótulos dentro."""
    path = os.path.join(OUT, name)
    if engine:
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02)
    fig.savefig(path)
    plt.close(fig)
    print("->", os.path.relpath(path, BASE))


# ------------------------------------------------------------------ arquitectura
def _box(ax, x, y, w, h, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=INK2, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8, color=INK, linespacing=1.15)


def _arrow(ax, p, q):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK2, shrinkA=0, shrinkB=0))


def arquitectura():
    fig, ax = plt.subplots(figsize=(FULL, 4.6 * CM))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    w, h, gap = 3.75, 1.35, 0.9
    xs = [0.2 + i * (w + gap) for i in range(4)]
    y1, y2 = 2.75, 0.35
    row1 = [("Video del celular\n1920×1080 a 30 FPS", "#e8f0fb"), ("YOLOv8l-seg\n12 clases, 640 px", "#e8f0fb"),
            ("ByteTrack\nidentidad por objeto", "#e8f0fb"), ("Distancia $D$\nEc. (1) y Ec. (2)", "#e9f6f0")]
    for x, (t, c) in zip(xs, row1):
        _box(ax, x, y1, w, h, t, c)
    for x0, x1 in zip(xs, xs[1:]):
        _arrow(ax, (x0 + w, y1 + h / 2), (x1, y1 + h / 2))
    row2 = [(xs[1], "Voz en español: peatón,\nreductor, semáforo rojo", "#fdf1e8"),
            (xs[2], "Gestor de prioridad\ny tiempo de espera", "#fdf1e8"),
            (xs[3], "TTC $= D/v$ y\nzona de riesgo", "#e9f6f0")]
    for x, t, c in row2:
        _box(ax, x, y2, w, h, t, c)
    _arrow(ax, (xs[3] + w / 2, y1), (xs[3] + w / 2, y2 + h))
    _arrow(ax, (xs[3], y2 + h / 2), (xs[2] + w, y2 + h / 2))
    _arrow(ax, (xs[2], y2 + h / 2), (xs[1] + w, y2 + h / 2))
    ax.text(xs[0], y1 + h + 0.18, "Percepción", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(xs[3], y1 + h + 0.18, "Estimación", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(xs[1], y2 + h + 0.18, "Alerta", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(xs[0], y2 + h / 2, "Interfaz: máscaras,\ndistancias y avisos.\nLaptop RTX 3050 (4 GB).",
            fontsize=7.5, color=INK2, va="center")
    save(fig, "fig_arquitectura.pdf", engine=False)


# ------------------------------------------------------------------ geometría pinhole
def pinhole():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 8.2 * CM))
    for ax in (a1, a2):
        ax.set_xlim(-2.2, 9.6)
        ax.set_ylim(-0.75, 2.75)
        ax.axis("off")
        ax.plot([-2.1, 9.5], [0, 0], color=INK, lw=0.9)
        for x in np.arange(-2.0, 9.5, 0.4):
            ax.plot([x, x - 0.15], [0, -0.15], color=GRID, lw=0.6)
    hc, D, H, f = 1.35, 7.8, 1.9, 1.3
    C = np.array([0.0, hc])
    # (a) altura conocida: triángulos semejantes, eje horizontal
    ax = a1
    ax.text(-2.1, 2.55, "Altura conocida, Ec. (1)", fontsize=8, color=INK)
    xp = -f
    for Y in (H, 0.0):
        yi = hc + (hc - Y) * f / D  # imagen invertida en el plano detrás del centro óptico
        ax.plot([D, xp], [Y, yi], color="#2a78d6" if Y else "#2a78d6", lw=0.7)
    y_top, y_bot = hc + (hc - H) * f / D, hc + hc * f / D
    ax.plot([xp, xp], [hc - 0.75, hc + 0.75], color=INK, lw=1.2)
    ax.plot([xp - 0.12, xp - 0.12], [y_top, y_bot], color="#eb6834", lw=2.2)
    ax.text(xp - 0.25, (y_top + y_bot) / 2, r"$h_\mathrm{px}$", ha="right", va="center", fontsize=8.5, color="#eb6834")
    ax.plot(*C, "o", ms=3.2, color=INK)
    ax.annotate("", xy=(0, hc - 0.9), xytext=(xp, hc - 0.9), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
    ax.text(xp / 2, hc - 0.85, r"$f$", ha="center", va="bottom", fontsize=8.5)
    ax.add_patch(plt.Rectangle((D - 0.1, 0), 0.2, H, fc="#2a78d6", ec="#2a78d6", lw=0.3, zorder=3))
    ax.text(D + 0.2, H / 2, r"$H$", va="center", fontsize=9, color="#2a78d6")
    ax.annotate("", xy=(D, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
    ax.text(D / 2, -0.42, r"$D = f\,H / h_\mathrm{px}$", ha="center", va="bottom", fontsize=8.5)
    ax.text(xp, hc + 0.85, "plano imagen", ha="center", fontsize=7.5, color=INK2)
    # (b) punto de contacto con el suelo y cámara inclinada
    ax = a2
    ax.text(-2.1, 2.55, "Punto de contacto con el suelo, Ec. (2)", fontsize=8, color=INK)
    th, Dg = np.deg2rad(6), 6.5
    ax.plot(*C, "o", ms=3.2, color=INK)
    ax.plot([0, 9.3], [hc, hc], ls=":", lw=0.6, color=INK2)
    ax.plot([0, 9.3], [hc, hc - 9.3 * np.tan(th)], ls="--", lw=0.7, color=INK2)
    ax.text(9.3, hc - 9.3 * np.tan(th) + 0.12, "eje óptico", fontsize=7.5, color=INK2, va="bottom", ha="right")
    ax.plot([0, Dg], [hc, 0], color="#eb6834", lw=0.9)
    ax.add_patch(plt.Rectangle((Dg - 0.1, 0), 0.2, 1.7, fc="#2a78d6", ec="#2a78d6", lw=0.3, zorder=3))
    ang = np.rad2deg(np.arctan2(hc, Dg))
    t = np.linspace(0, np.deg2rad(ang), 30)
    ax.plot(2.2 * np.cos(t), hc - 2.2 * np.sin(t), color="#eb6834", lw=0.6)
    ax.text(2.35, hc - 0.28, r"$\alpha$", fontsize=8.5, color="#eb6834")
    t2 = np.linspace(0, th, 20)
    ax.plot(3.6 * np.cos(t2), hc - 3.6 * np.sin(t2), color=INK2, lw=0.6)
    ax.text(3.75, hc - 0.02, r"$\theta$", fontsize=8.5, va="center")
    ax.annotate("", xy=(-0.35, 0.0), xytext=(-0.35, hc), arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK,
                shrinkA=0, shrinkB=0))
    ax.plot([-0.5, 0.0], [hc, hc], lw=0.5, color=INK2)
    ax.text(-0.5, hc / 2, r"$h_\mathrm{cam}$", ha="right", va="center", fontsize=8.5)
    ax.annotate("", xy=(Dg, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
    ax.text(Dg / 2, -0.42, r"$D = h_\mathrm{cam} / \tan\alpha$", ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(hspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_pinhole.pdf", engine=False)


# ------------------------------------------------------------------ foto del montaje
def montaje():
    src = os.path.join(BASE, "OBSERVACIONES", "FOTO PARA REVISOR 1.jpeg")
    im = Image.open(src).convert("RGB")
    W, H = im.size
    im = im.crop((int(0.02 * W), 0, W, int(0.93 * H)))
    target_w = int(8.5 / 2.54 * 300)
    im = im.resize((target_w, int(im.height * target_w / im.width)), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("times.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    s = im.width / (W * 0.98)

    def label(xy_box, xy_text, text):
        (bx, by), (tx, ty) = [(int(x * s), int(y * s)) for x, y in (xy_box, xy_text)]
        dr.line([(tx, ty), (bx, by)], fill=(255, 255, 255), width=5)
        dr.line([(tx, ty), (bx, by)], fill=(20, 20, 20), width=2)
        tb = dr.textbbox((tx, ty), text, font=font)
        if tb[2] > im.width - 12:  # no cabe: se corre a la izquierda
            tx -= tb[2] - (im.width - 12)
            tb = dr.textbbox((tx, ty), text, font=font)
        dr.rectangle([tb[0] - 8, tb[1] - 6, tb[2] + 8, tb[3] + 6], fill=(255, 255, 255))
        dr.text((tx, ty), text, fill=(10, 10, 10), font=font)
    label((925, 560), (1080, 900), "soporte con ventosa")
    label((700, 470), (560, 90), "vía al frente")
    path = os.path.join(OUT, "fig_montaje_mototaxi.png")
    im.save(path, dpi=(300, 300))
    print("->", os.path.relpath(path, BASE))


# ------------------------------------------------------------------ distribución antes/después
def distribucion():
    """Instancias por clase antes (corpus original) y después (más la recolección dirigida): puntos unidos por
    una flecha sobre eje logarítmico, sin barras; los conteos van en dos columnas fuera del área de datos."""
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    d1 = np.array([sum(int(r[f"c{k}"]) for r in rows if r["fuente"] in ("L", "A")) for k in range(12)])
    tot = np.array([sum(int(r[f"c{k}"]) for r in rows) for k in range(12)])
    order = np.argsort(tot)
    fig, ax = plt.subplots(figsize=(COL, 6.4 * CM))
    fig.subplots_adjust(left=0.30, right=0.64, top=0.84, bottom=0.17)
    n_d1 = sum(r["fuente"] in ("L", "A") for r in rows)
    for i, k in enumerate(order):
        x0 = d1[k] if d1[k] > 0 else 1.0
        ax.annotate("", xy=(tot[k], i), xytext=(x0, i),
                    arrowprops=dict(arrowstyle="-|>", color="#b9b7b0" if d1[k] else INK2, lw=0.9, shrinkA=4, shrinkB=4), zorder=1)
        ax.plot(x0, i, SERIES["E1R"][1], ms=4.6, color=SERIES["E1R"][0], mfc=SERIES["E1R"][0] if d1[k] else "white", zorder=3)
        ax.plot(tot[k], i, SERIES["E2"][1], ms=4.8, color=SERIES["E2"][0], zorder=3)
    ax.set_yticks(range(12), [SHORT[k] for k in order])
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.8, 3000)
    ax.set_ylim(-0.6, 11.6)
    ax.set_xlabel("Instancias (escala logarítmica)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tr = ax.get_yaxis_transform()
    ax.text(1.24, 11.9, "Antes", transform=tr, ha="right", fontsize=7.5, color=INK2, clip_on=False)
    ax.text(1.58, 11.9, "Después", transform=tr, ha="right", fontsize=7.5, color=INK2, clip_on=False)
    for i, k in enumerate(order):
        ax.text(1.24, i, str(d1[k]), transform=tr, ha="right", va="center", fontsize=7.5, color=INK2, clip_on=False)
        ax.text(1.58, i, str(tot[k]), transform=tr, ha="right", va="center", fontsize=7.5, color=INK, clip_on=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=SERIES["E1R"][1], color=SERIES["E1R"][0], ls="", ms=4.6, label=f"Corpus original ({n_d1} imágenes)"),
                       Line2D([], [], marker=SERIES["E2"][1], color=SERIES["E2"][0], ls="", ms=4.8, label=f"Corpus final ({len(rows)} imágenes)")],
              loc="lower left", bbox_to_anchor=(-0.45, 1.03), ncol=1, frameon=False, handletextpad=0.3)
    save(fig, "fig_distribucion.pdf", engine=False)


# ------------------------------------------------------------------ AP por clase (panel b del desbalance)
def ap_por_clase(kind="box"):
    res = json.load(open(os.path.join(HERE, "runs", "r1", "resumen_resultados.json"), encoding="utf-8"))
    main = res["subconjuntos"]["principal"]
    ics = res["bootstrap"]["ap_clase_ic95"]
    est = set(res["clases_estimables"])
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    tot = np.array([sum(int(r[f"c{k}"]) for r in rows) for k in range(12)])
    order = np.argsort(tot)  # mismo orden que el panel (a)
    fig, ax = plt.subplots(figsize=(COL, 6.4 * CM))
    offsets = {"E0": 0.24, "E1R": 0.0, "E2": -0.24}
    names = {"E0": "E0: sin ajuste", "E1R": "E1: corpus original", "E2": "E2: + recolección dirigida"}
    for m in ("E0", "E1R", "E2"):
        if m not in main:
            continue
        col, mk = SERIES[m]
        for i, c in enumerate(order):
            v = main[m]["por_clase"][kind].get(str(c), main[m]["por_clase"][kind].get(c))
            if c not in est or v is None:
                continue
            ci = ics.get(m, {}).get(kind, {}).get(str(c)) or ics.get(m, {}).get(kind, {}).get(c)
            y = i + offsets[m]
            if ci:
                ax.plot(ci, [y, y], color=col, lw=0.9, alpha=0.8)
            ax.plot(v, y, mk, color=col, ms=4.2, mec="white", mew=0.5,
                    label=names[m] if i == 0 else None)
    for i, c in enumerate(order):
        if c not in est:
            ax.text(0.02, i, "no estimable (< 3 grupos)", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(12), [SHORT[k] for k in order])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 11.6)
    ax.set_xlabel("AP50 de " + ("caja" if kind == "box" else "máscara") + " en prueba (IC 95 %)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, ncol=1, handletextpad=0.3)
    save(fig, f"fig_ap_clase_{kind}.pdf")


# ------------------------------------------------------------------ ejemplos cualitativos E1 vs E2 (clases escasas)
RARE_COL = {0: (235, 104, 52), 1: (27, 175, 122), 2: (42, 120, 214), 3: (237, 161, 0)}


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def ejemplos(n=3, device="cpu"):
    """Escenas de prueba (sin Street View) con más elementos de la calzada anotados, una por grupo (criterio
    neutro, no elegido por el resultado). Filas: E0, E1 y E2 con confianza 0.35; solo máscaras de las clases 0-3.
    Se difuminan las cabezas de todas las personas anotadas y detectadas."""
    from ultralytics import YOLO
    from PIL import ImageFilter
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
            if r["split"] == "test" and r["fuente"] != "SV" and sum(int(r[f"c{k}"]) for k in range(4)) > 0]
    rows.sort(key=lambda r: (-sum(int(r[f"c{k}"]) for k in range(4)), r["uid"]))
    chosen, groups = [], set()
    for r in rows:
        if r["grupo"] in groups:
            continue
        chosen.append(r); groups.add(r["grupo"])
        if len(chosen) == n:
            break
    models = [("E0", os.path.join(BASE, "best.pt")),
              ("E1", os.path.join(HERE, "runs", "r1", "E1R_s0", "weights", "last.pt")),
              ("E2", os.path.join(HERE, "runs", "r1", "E2_s0", "weights", "last.pt"))]
    preds = {}
    for name, w in models:
        m = YOLO(w)
        for r in chosen:
            img = os.path.join(HERE, "dataset_v2", "images", r["archivo"])
            res = m.predict(img, imgsz=640, conf=0.35, verbose=False, device=device, retina_masks=True)[0]
            P = []
            if res.masks is not None:
                for poly, b, c in zip(res.masks.xy, res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
                    P.append((int(c), b, poly))
            preds[(name, r["uid"])] = P
    W = int(18 / 2.54 * 300); gap = 14
    tile_w = (W - 60 - (n - 1) * gap) // n; tile_h = int(tile_w * 0.56); lab_h = 46
    top = 100
    canvas = Image.new("RGB", (W, top + len(models) * (tile_h + lab_h)), "white")
    try:
        font = ImageFont.truetype("times.ttf", 34); fontb = ImageFont.truetype("timesbd.ttf", 36)
    except OSError:
        font = fontb = ImageFont.load_default()
    td = ImageDraw.Draw(canvas)
    for j, r in enumerate(chosen):
        img = os.path.join(HERE, "dataset_v2", "images", r["archivo"])
        im = Image.open(img).convert("RGB")
        d = json.load(open(os.path.join(HERE, "dataset_v2", "unified_json", r["uid"] + ".json"), encoding="utf-8"))
        heads, gts = [], []
        for sh in d["shapes"]:
            xs, ys = zip(*sh["points"]); bb = [min(xs), min(ys), max(xs), max(ys)]
            k = SHORT_ID.get(sh["label"])
            if k is not None:
                gts.append(bb)
            if sh["label"] == "Persona":
                heads.append(bb)
            if sh["label"] in VEHICULOS and bb[2] - bb[0] > 150:  # placa: franja trasera central
                w_, h_ = bb[2] - bb[0], bb[3] - bb[1]
                pb = (int(bb[0] + 0.2 * w_), int(bb[1] + 0.5 * h_), int(bb[0] + 0.8 * w_), int(bb[1] + 0.9 * h_))
                im.paste(im.crop(pb).filter(ImageFilter.GaussianBlur(7)), pb[:2])
        for name, _ in models:
            heads += [b for c, b, _ in preds[(name, r["uid"])] if c == 4]
        for b in heads:  # difuminar el tercio superior de cada persona (rostro)
            hb = (int(b[0]), int(b[1]), int(b[2]), int(b[1] + 0.3 * (b[3] - b[1])) + 1)
            if hb[2] > hb[0] and hb[3] > hb[1]:
                im.paste(im.crop(hb).filter(ImageFilter.GaussianBlur(10)), hb[:2])
        ub = [min(g[0] for g in gts), min(g[1] for g in gts), max(g[2] for g in gts), max(g[3] for g in gts)]
        cw = min(im.width, max(ub[2] - ub[0] + 160, im.width * 0.6)); ch = cw * tile_h / tile_w
        cx, cy = (ub[0] + ub[2]) / 2, (ub[1] + ub[3]) / 2
        x0 = min(max(0, cx - cw / 2), im.width - cw); y0 = min(max(0, cy - ch / 2), im.height - ch)
        box = (int(x0), int(max(0, y0)), int(x0 + cw), int(max(0, y0) + ch))
        for row, (name, _) in enumerate(models):
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            P = preds[(name, r["uid"])]
            for c, b, poly in P:
                if c in RARE_COL and len(poly) >= 3:
                    od.polygon([tuple(p) for p in poly], fill=RARE_COL[c] + (120,), outline=RARE_COL[c] + (255,))
            base = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            tile = base.crop(box).resize((tile_w, tile_h), Image.LANCZOS)
            X = 60 + j * (tile_w + gap); Y = top + row * (tile_h + lab_h)
            canvas.paste(tile, (X, Y))
            cnt = collections.Counter(c for c, _, _ in P if c in RARE_COL)
            txt = "   ".join(f"{TINY[k]} {cnt[k]}" for k in sorted(cnt)) if cnt else "ninguno detectado"
            td.text((X + 4, Y + tile_h + 6), txt, fill=(10, 10, 10), font=font)
            if j == 0:
                td.text((6, Y + tile_h // 2 - 18), name, fill=(10, 10, 10), font=fontb)
        td.text((60 + j * (tile_w + gap), top - 42), f"({'abc'[j]}) {int(sum(int(r[f'c{k}']) for k in range(4)))} elementos anotados",
                fill=(10, 10, 10), font=font)
    lx = 60
    for k in (2, 3, 1, 0):  # leyenda de colores
        td.rectangle((lx, 12, lx + 30, 42), fill=RARE_COL[k])
        td.text((lx + 40, 8), LEG[k], fill=(10, 10, 10), font=font)
        lx += 40 + int(td.textlength(LEG[k], font=font)) + 50
    path = os.path.join(OUT, "fig_ejemplos.png")
    canvas.save(path, dpi=(300, 300))
    print("->", os.path.relpath(path, BASE), [r["uid"] for r in chosen])


VEHICULOS = {"Automóvil particular", "Camión", "Bus de transporte", "Mototaxi", "Motocicleta"}
TINY = {0: "reductor", 1: "paso", 2: "recta", 3: "recta y der."}
LEG = {0: "Reductor", 1: "Paso peatonal", 2: "Flecha recta", 3: "Flecha recta y derecha"}
SHORT_ID = {"Reductor de velocidad": 0, "Paso peatonal": 1, "Línea recta": 2, "Línea recta y derecha": 3}


# ------------------------------------------------------------------ sensibilidad de la distancia (presupuesto de error)
SENS = dict(f=950.0, H=1.65, sH=0.07, sh=2.0, h_cam=1.40, s_theta_deg=(1.0, 0.5), s_y=2.0)


def sensibilidad():
    """Error esperado (1 desviación estándar) de cada ecuación en función de la distancia.
    Ec. (1): sigma = D * sqrt((sH/H)^2 + (sh*D/(f*H))^2)
    Ec. (2): sigma = ((D^2 + h_cam^2)/h_cam) * sqrt(s_theta^2 + (s_y/f)^2)   (derivada exacta de h_cam/tan(alpha))"""
    p = SENS
    D = np.linspace(0.3, 25, 300)
    s1 = D * np.sqrt((p["sH"] / p["H"]) ** 2 + (p["sh"] * D / (p["f"] * p["H"])) ** 2)
    fig, ax = plt.subplots(figsize=(COL, 5.6 * CM))
    ax.axvspan(0, 3, color="#fbe3d6", lw=0)
    ax.axvspan(3, 7, color="#fdf1e8", lw=0)
    ax.text(0.25, 5.55, "peligro", fontsize=7.5, color=INK2)
    ax.text(3.25, 5.55, "precaución", fontsize=7.5, color=INK2)
    ax.plot(D, s1, color=SERIES["E2"][0], lw=1.6, label="Ec. (1), altura conocida")
    for sd, ls in zip(p["s_theta_deg"], ("-", "--")):
        s2 = ((D ** 2 + p["h_cam"] ** 2) / p["h_cam"]) * np.sqrt(np.radians(sd) ** 2 + (p["s_y"] / p["f"]) ** 2)
        ax.plot(D, s2, color=SERIES["E1R"][0], lw=1.6, ls=ls,
                label=f"Ec. (2), inclinación ±{sd:g}°")
    ax.axvline(19.4, color=INK, lw=1.0, ls=(0, (2, 2)), zorder=3)
    ax.text(19.1, 5.55, "detención a 30 km/h", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Distancia real $D$ (m)")
    ax.set_ylabel("Error esperado (m)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=True, framealpha=0.95, edgecolor="none")
    save(fig, "fig_sensibilidad_distancia.pdf")
    # cifras para el texto
    def s2_at(d, sd):
        return ((d ** 2 + p["h_cam"] ** 2) / p["h_cam"]) * math.hypot(math.radians(sd), p["s_y"] / p["f"])

    def s1_at(d):
        return d * math.hypot(p["sH"] / p["H"], p["sh"] * d / (p["f"] * p["H"]))
    for d in (3, 7, 10, 20):
        print(f"D={d:>2} m  Ec1={s1_at(d):.2f}  Ec2(1°)={s2_at(d,1):.2f}  Ec2(0.5°)={s2_at(d,0.5):.2f}")
    for sd in p["s_theta_deg"]:
        grid = np.linspace(1, 25, 2401)
        cross = grid[np.argmin(np.abs([s1_at(g) - s2_at(g, sd) for g in grid]))]
        print(f"cruce Ec1=Ec2 con inclinación ±{sd}°: {cross:.1f} m")


# ------------------------------------------------------------------ resumen de confusión (legible a una columna)
def confusion(name="E2_s0_test_sinSV"):
    """Barra apilada por clase real: detectada correctamente, confundida con otra clase, omitida (fondo).
    Matriz de Ultralytics: filas = predicción, columnas = verdad; última fila/columna = fondo."""
    M = np.load(os.path.join(HERE, "runs", "r1", "eval", f"{name}_confusion.npy"))
    nc = M.shape[0] - 1
    cols = M[:, :nc]
    support = cols.sum(0)
    keep = [c for c in range(nc) if support[c] > 0]
    order = sorted(keep, key=lambda c: cols[c, c] / support[c])
    fig, ax = plt.subplots(figsize=(COL, 6.0 * CM))
    for i, c in enumerate(order):
        s = support[c]
        ok, miss = cols[c, c] / s, cols[nc, c] / s
        conf = 1 - ok - miss
        ax.barh(i, ok, color=SERIES["E2"][0], height=0.62, edgecolor="white", linewidth=0.5)
        ax.barh(i, conf, left=ok, color=SERIES["E1R"][0], height=0.62, edgecolor="white", linewidth=0.5)
        ax.barh(i, miss, left=ok + conf, color=SERIES["E0"][0], height=0.62, edgecolor="white", linewidth=0.5)
        ax.text(ok / 2 if ok > 0.12 else ok + 0.01, i, f"{ok*100:.0f}", va="center",
                ha="center" if ok > 0.12 else "left", fontsize=7.5, color="white" if ok > 0.12 else INK)
        other = [(cols[p, c], p) for p in range(nc) if p != c and cols[p, c] > 0]
        if other and conf >= 0.05:
            v, p = max(other)
            ax.text(1.02, i, f"→ {SHORT[p]}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), [f"{SHORT[c]} ({int(support[c])})" for c in order])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100 %"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES["E2"][0], label="correcta"), Patch(color=SERIES["E1R"][0], label="otra clase"),
                       Patch(color=SERIES["E0"][0], label="omitida (fondo)")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, handlelength=1.0, columnspacing=0.8)
    save(fig, "fig_confusion.pdf")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "todas"
    fns = dict(arquitectura=arquitectura, pinhole=pinhole, montaje=montaje, distribucion=distribucion,
               ap_clase=ap_por_clase, confusion=confusion, sensibilidad=sensibilidad, ejemplos=ejemplos)
    for k, fn in fns.items():
        if which in (k, "todas"):
            fn()
