# -*- coding: utf-8 -*-
"""Resultados R1 a partir de las evaluaciones guardadas (runs/r1/eval/*.pkl).

- Métricas en el test principal (sin Street View), por fuente y en Street View.
- Media y desviación estándar entre semillas.
- Bootstrap pareado por grupos de test (2000 remuestreos): IC 95 % de las diferencias entre modelos
  y del AP por clase (promediando las semillas dentro de cada remuestreo).
- AP 'no estimable' cuando la clase aparece en menos de 3 grupos del test principal.

Salidas: runs/r1/resumen_resultados.json y fragmentos LaTeX en RICUY_ADAS_UNI_2026_R1/tablas/.
Uso: python r1_results.py [--n 2000]
"""
import os, sys, json, glob, pickle, csv, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r1_eval import map_from_stats, EVAL_DIR, MANIFEST
from labelme2yolo_best import CLASSES

TAB_DIR = os.path.join(os.path.dirname(HERE), "RICUY_ADAS_UNI_2026_R1", "tablas")
OUT_JSON = os.path.join(HERE, "runs", "r1", "resumen_resultados.json")
SHORT = ["Reductor de velocidad", "Paso peatonal", "Flecha recta", "Flecha recta y derecha", "Persona",
         "Motocicleta", "Automóvil particular", "Camión", "Bus de transporte", "Semáforo en verde",
         "Semáforo en rojo", "Mototaxi"]
MODELS = {"E0": ["E0_640"], "E0_1024": ["E0_1024"], "E1R": ["E1R_s0", "E1R_s1", "E1R_s2"],
          "E2": ["E2_s0", "E2_s1", "E2_s2"], "E2noSV": ["E2noSV_s0", "E2noSV_s1", "E2noSV_s2"],
          "E2_1024": ["E2_s0_1024", "E2_s1_1024", "E2_s2_1024"],
          "E1Rc": ["E1Rc_s0"], "E2c": ["E2c_s0"]}  # sensibilidad: señales de cruce enmascaradas en D1
# control dejando fuera el recorrido del 12/09 (fuente O): evaluados solo en sus fotogramas de prueba
MODELS_SRCO = {"E0": ["E0_640_srcO"], "E1R": ["E1R_s0_srcO", "E1R_s1_srcO", "E1R_s2_srcO"],
               "E2": ["E2_s0_srcO", "E2_s1_srcO", "E2_s2_srcO"], "E2noO": ["E2noO_s0_srcO"]}
SEMAFOROS = (9, 10)
KEYS = ["box50", "box5095", "mask50", "mask5095"]


def load(models=MODELS):
    runs = {}
    for m, names in models.items():
        got = []
        for n in names:
            p = os.path.join(EVAL_DIR, f"{n}_test.pkl")
            if os.path.exists(p):
                got.append(pickle.load(open(p, "rb")))
        if got:
            runs[m] = got
    return runs


def manifest():
    info = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                info[r["archivo"]] = dict(grupo=r["grupo"], fuente=r["fuente"],
                                          c=np.array([int(r[f"c{k}"]) for k in range(12)]))
    return info


def idx_of(per_image):
    return {r["im_name"]: i for i, r in enumerate(per_image)}


def metrics_on(per_image_list, names):
    """Media y DE entre semillas de las métricas globales y AP por clase sobre un subconjunto de imágenes."""
    vals = collections.defaultdict(list)
    cls = collections.defaultdict(lambda: collections.defaultdict(list))
    for pi in per_image_list:
        ix = idx_of(pi)
        m = map_from_stats(pi, [ix[n] for n in names if n in ix])
        for k in KEYS:
            vals[k].append(m[k])
        for kind in ("box", "mask"):
            for c, ap in m[f"{kind}_ap50_cls"].items():
                cls[kind][c].append(ap)
    out = {k: dict(media=float(np.mean(v)), de=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))
           for k, v in vals.items()}
    out["por_clase"] = {kind: {int(c): float(np.mean(v)) for c, v in d.items()} for kind, d in cls.items()}
    out["por_clase_semillas"] = {kind: {int(c): [float(x) for x in v] for c, v in d.items()} for kind, d in cls.items()}
    # sensibilidad: mAP50 sin las dos clases de semáforo (la convención de anotación cambió en la revisión)
    for kind in ("box", "mask"):
        per = [np.mean([ap for c, ap in cls_seed.items() if c not in SEMAFOROS])
               for cls_seed in _per_seed_cls(per_image_list, names, kind)]
        out[f"{kind}50_sin_semaforos"] = dict(media=float(np.mean(per)),
                                              de=float(np.std(per, ddof=1)) if len(per) > 1 else None)
    return out


def _per_seed_cls(per_image_list, names, kind):
    res = []
    for pi in per_image_list:
        ix = idx_of(pi)
        m = map_from_stats(pi, [ix[n] for n in names if n in ix])
        res.append(m[f"{kind}_ap50_cls"])
    return res


def bootstrap(runs, names, groups_of, n_boot=2000, seed=0, pairs=(), classes_for=("E0", "E1R", "E2")):
    rng = np.random.default_rng(seed)
    by_g = collections.defaultdict(list)
    for nm in names:
        by_g[groups_of[nm]].append(nm)
    G = sorted(by_g)
    idx = {m: [idx_of(pi) for pi in lst] for m, lst in runs.items()}
    diffs = {f"{a}-{b}": {k: [] for k in ("box50", "mask50", "box50_sin_sem")} for a, b in pairs}
    cls_samples = {m: {kind: collections.defaultdict(list) for kind in ("box", "mask")} for m in classes_for if m in runs}
    for _ in range(n_boot):
        pick = rng.choice(len(G), len(G), replace=True)
        ks = [nm for gi in pick for nm in by_g[G[gi]]]
        cache = {}
        for m in set([x for p in pairs for x in p]) | set(cls_samples):
            if m not in runs:
                continue
            per_seed = [map_from_stats(pi, [ix[k] for k in ks if k in ix]) for pi, ix in zip(runs[m], idx[m])]
            cache[m] = per_seed
        for a, b in pairs:
            if a in cache and b in cache:
                for k in ("box50", "mask50"):
                    diffs[f"{a}-{b}"][k].append(np.mean([s[k] for s in cache[a]]) - np.mean([s[k] for s in cache[b]]))
                sin = lambda m: np.mean([np.mean([ap for c, ap in s["box_ap50_cls"].items() if c not in SEMAFOROS])
                                         for s in cache[m]])
                diffs[f"{a}-{b}"]["box50_sin_sem"].append(sin(a) - sin(b))
        for m in cls_samples:
            for kind in ("box", "mask"):
                acc = collections.defaultdict(list)
                for s in cache[m]:
                    for c, ap in s[f"{kind}_ap50_cls"].items():
                        acc[c].append(ap)
                for c, v in acc.items():
                    cls_samples[m][kind][c].append(np.mean(v))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    dif_cls = {}
    for a, b in (("E2", "E0"), ("E2", "E1R"), ("E1R", "E0")):
        if a in cls_samples and b in cls_samples:
            dif_cls[f"{a}-{b}"] = {int(c): ci(np.array(cls_samples[a]["box"][c]) - np.array(cls_samples[b]["box"][c]))
                                   for c in cls_samples[a]["box"]
                                   if len(cls_samples[a]["box"][c]) == len(cls_samples[b]["box"].get(c, []))}
    res = {"ap_clase_dif_ic95": dif_cls,
           "diferencias": {p: {k: dict(ic95=ci(v)) for k, v in d.items()} for p, d in diffs.items()},
           "ap_clase_ic95": {m: {kind: {int(c): ci(v) for c, v in d.items()} for kind, d in kk.items()}
                             for m, kk in cls_samples.items()},
           "n_grupos": len(G), "n_imagenes": len(names), "n_boot": n_boot}
    return res


def latex_tables(summary):
    os.makedirs(TAB_DIR, exist_ok=True)
    main = summary["subconjuntos"]["principal"]
    rows = []
    label = {"E0": "E0, modelo de QHAWAY sin ajuste & 640", "E0_1024": "E0, modelo de QHAWAY sin ajuste & 1024",
             "E1R": "E1, ajuste con D1 & 640", "E2": "E2, ajuste con D1 y D2 & 640",
             "E2noSV": "E2-SV, E2 sin Street View & 640", "E2_1024": "E2, inferencia a 1024 píxeles & 1024"}
    fmt = lambda d: (f"{d['media']:.3f}" + (f" $\pm$ {d['de']:.3f}" if d.get("de") else ""))
    for m in ["E0", "E0_1024", "E1R", "E2", "E2_1024", "E2noSV"]:
        if m in main:
            r = main[m]
            rows.append(f"{label[m]} & {fmt(r['box50'])} & {fmt(r['box5095'])} & {fmt(r['mask50'])} & "
                        f"{fmt(r['mask5095'])} & {fmt(r['box50_sin_semaforos'])} \\\\")
    with open(os.path.join(TAB_DIR, "tab_global.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    # AP50 de caja por clase: E0, E1, E2 con IC de E2
    est = summary["clases_estimables"]
    ics = summary["bootstrap"]["ap_clase_ic95"]
    lines = []
    for c in range(12):
        n = summary["instancias_test_principal"][c]
        ng = summary["grupos_test_principal"][c]
        cells = []
        for m in ("E0", "E1R", "E2"):
            v = main.get(m, {}).get("por_clase", {}).get("box", {}).get(c)
            cells.append("n.~e." if (c not in est or v is None) else f"{v:.2f}")
        ic = ics.get("E2", {}).get("box", {}).get(c)
        ic_txt = f"[{ic[0]:.2f}, {ic[1]:.2f}]" if (ic and c in est) else "n.~e."
        lines.append(f"{SHORT[c]} & {n} ({ng}) & {' & '.join(cells)} & {ic_txt} \\\\")
    with open(os.path.join(TAB_DIR, "tab_por_clase.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


TAB_SHORT = ["Reductor", "Paso peatonal", "Flecha recta", "Flecha recta y der.", "Persona", "Motocicleta",
             "Automóvil", "Camión", "Bus", "Semáforo verde", "Semáforo rojo", "Mototaxi"]


def cifras_dataset():
    """Composición del corpus y de la partición (desde split_manifest.csv) y la Tabla de partición."""
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    C = lambda r: np.array([int(r[f"c{k}"]) for k in range(12)])
    d = {}
    D1 = [r for r in rows if r["fuente"] in ("L", "A")]
    D2 = [r for r in rows if r["fuente"] not in ("L", "A")]
    tot = sum(C(r) for r in rows)
    d1 = sum(C(r) for r in D1)
    d["ds@img@total"], d["ds@img@D1"], d["ds@img@D2"] = str(len(rows)), str(len(D1)), str(len(D2))
    d["ds@img@L"] = str(sum(r["fuente"] == "L" for r in rows))
    d["ds@img@A"] = str(sum(r["fuente"] == "A" for r in rows))
    d["ds@img@SV"] = str(sum(r["fuente"] == "SV" for r in rows))
    d["ds@inst@total"], d["ds@inst@D1"] = str(int(tot.sum())), str(int(d1.sum()))
    d["ds@inst@D2"] = str(int(tot.sum() - d1.sum()))
    d["ds@inst@SV"] = str(int(sum(C(r).sum() for r in rows if r["fuente"] == "SV")))
    d["ds@grupos"] = str(len({r["grupo"] for r in rows}))
    for c in range(12):
        d[f"ds@cls@{c}@total"], d[f"ds@cls@{c}@D1"] = str(int(tot[c])), str(int(d1[c]))
    d["ds@ratio@D1"] = f"{d1[6] / max(d1[0], 1):.0f}"
    d["ds@ratio@final"] = f"{tot[6] / tot[0]:.0f}"
    d["ds@d2@auto"], d["ds@d2@persona"] = str(int(tot[6] - d1[6])), str(int(tot[4] - d1[4]))
    d["ds@d2@calzada"] = str(int(sum(tot[k] - d1[k] for k in range(4))))
    lines = []
    for sname in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == sname]
        d[f"ds@img@{sname}"] = str(len(rs))
        d[f"ds@grp@{sname}"] = str(len({r["grupo"] for r in rs}))
        d[f"ds@inst@{sname}"] = str(int(sum(C(r).sum() for r in rs)))
    main = [r for r in rows if r["split"] == "test" and r["fuente"] != "SV"]
    # semáforos de la prueba principal cuyo contorno proviene del modelo base (armonización)
    prop = json.load(open(os.path.join(HERE, "dataset_v2", "semaforos_propuesta.json"), encoding="utf-8"))
    firma = {json.dumps(x["points"]) for v in prop.values() for x in (v["nuevos"] or [])}
    n_e0 = n_sem = 0
    for r in main:
        js = json.load(open(os.path.join(HERE, "dataset_v2", "unified_json", r["uid"] + ".json"), encoding="utf-8"))
        for sh in js["shapes"]:
            if sh["label"] in (CLASSES[9], CLASSES[10]):
                n_sem += 1
                n_e0 += json.dumps(sh["points"]) in firma
    d["sem@test@e0"], d["sem@test@total"] = str(n_e0), str(n_sem)
    d["ds@img@main"], d["ds@grp@main"] = str(len(main)), str(len({r["grupo"] for r in main}))
    d["ds@inst@main"] = str(int(sum(C(r).sum() for r in main)))
    for c in range(12):
        cells = []
        for sub in ("train", "val", "test", "main"):
            rs = main if sub == "main" else [r for r in rows if r["split"] == sub]
            n = int(sum(C(r)[c] for r in rs))
            g = len({r["grupo"] for r in rs if C(r)[c] > 0})
            cells.append(f"{n} ({g})")
        lines.append(f"{TAB_SHORT[c]} & " + " & ".join(cells) + r" \\")
    os.makedirs(TAB_DIR, exist_ok=True)
    open(os.path.join(TAB_DIR, "tab_particion.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return d


def cifras_tex(summary):
    """Cada cifra del texto sale de aquí: tablas/cifras.tex define las claves que lee \\cf{clave} en Barboza.tex."""
    d = {}
    main = summary["subconjuntos"]["principal"]
    for m, r in main.items():
        if m == "n_imagenes":
            continue
        for k in KEYS:
            d[f"g@{m}@{k}"] = f"{r[k]['media']:.3f}"
            if r[k].get("de") is not None:
                d[f"gde@{m}@{k}"] = f"{r[k]['de']:.3f}"
        for kind in ("box", "mask"):
            d[f"gsin@{m}@{kind}"] = f"{r[f'{kind}50_sin_semaforos']['media']:.3f}"
            for c, v in r["por_clase"][kind].items():
                d[f"c@{m}@{kind}@{c}"] = f"{v:.2f}"
            for c, v in r["por_clase_semillas"][kind].items():
                if len(v) > 1:
                    d[f"cmin@{m}@{kind}@{c}"] = f"{min(v):.2f}"
                    d[f"cmax@{m}@{kind}@{c}"] = f"{max(v):.2f}"
    for m, kk in summary["bootstrap"]["ap_clase_ic95"].items():
        for kind, dd in kk.items():
            for c, ic in dd.items():
                if ic:
                    d[f"cilo@{m}@{kind}@{c}"], d[f"cihi@{m}@{kind}@{c}"] = f"{ic[0]:.2f}", f"{ic[1]:.2f}"
    for pair, dd in summary["bootstrap"]["diferencias"].items():
        a, b = pair.split("-")
        for k, v in dd.items():
            if v["ic95"]:
                ka = "box50_sin_semaforos" if k == "box50_sin_sem" else k
                d[f"d@{pair}@{k}"] = f"${main[a][ka]['media'] - main[b][ka]['media']:+.3f}$"
                d[f"dlo@{pair}@{k}"], d[f"dhi@{pair}@{k}"] = f"${v['ic95'][0]:.3f}$", f"${v['ic95'][1]:.3f}$"
    for pair, dd in summary["bootstrap"].get("ap_clase_dif_ic95", {}).items():
        a, b = pair.split("-")
        for c, ic in dd.items():
            if ic:
                d[f"dc@{pair}@{c}"] = f"${main[a]['por_clase']['box'][c] - main[b]['por_clase']['box'][c]:+.2f}$"
                d[f"dclo@{pair}@{c}"], d[f"dchi@{pair}@{c}"] = f"${ic[0]:.2f}$", f"${ic[1]:.2f}$"
    for sub, dd in summary["subconjuntos"].items():
        if sub == "principal":
            continue
        for m, r in dd.items():
            if m != "n_imagenes":
                d[f"s@{sub}@{m}@box50"] = f"{r['box50']['media']:.3f}"
                d[f"s@{sub}@{m}@mask50"] = f"{r['mask50']['media']:.3f}"
                for c, v in r["por_clase"]["box"].items():
                    d[f"s@{sub}@{m}@box@{c}"] = f"{v:.2f}"
        d[f"s@{sub}@nimg"] = str(dd["n_imagenes"])
    co = summary.get("control_recorrido_O")
    if co:
        d["o@nimg"], d["o@ngrupos"] = str(co["n_imagenes"]), str(co["n_grupos"])
        for c, n in enumerate(co["instancias"]):
            d[f"o@inst@{c}"] = str(n)
        for m, r in co["modelos"].items():
            d[f"o@{m}@box50"] = f"{r['box50']['media']:.3f}"
            d[f"o@{m}@mask50"] = f"{r['mask50']['media']:.3f}"
            for c, v in r["por_clase"]["box"].items():
                d[f"o@{m}@box@{c}"] = f"{v:.2f}"
            for c, v in r["por_clase_semillas"]["box"].items():
                if len(v) > 1:
                    d[f"omin@{m}@box@{c}"], d[f"omax@{m}@box@{c}"] = f"{min(v):.2f}", f"{max(v):.2f}"
    for c in range(12):
        d[f"n@inst@{c}"] = str(summary["instancias_test_principal"][c])
        d[f"n@grp@{c}"] = str(summary["grupos_test_principal"][c])
    d["n@img"] = str(summary["subconjuntos"]["principal"]["n_imagenes"])
    palabra = {1: "una", 2: "dos", 3: "tres"}
    for m, k in summary["modelos_disponibles"].items():
        d[f"nsem@{m}"] = palabra.get(k, str(k))
    d["n@grupos"] = str(summary["bootstrap"]["n_grupos"])
    d["n@boot"] = str(summary["bootstrap"]["n_boot"])
    for sz in (640, 1024):
        lp = os.path.join(HERE, "runs", "r1", f"latencia_{sz}.json")
        if os.path.exists(lp):
            L = json.load(open(lp, encoding="utf-8"))
            for prec in ("FP16", "FP32"):
                if prec in L:
                    x = L[prec]
                    d[f"lat@{sz}@{prec}@pre"] = f"{x['pre_ms']:.1f}"
                    d[f"lat@{sz}@{prec}@inf"] = f"{x['inferencia_ms']:.1f}"
                    d[f"lat@{sz}@{prec}@post"] = f"{x['post_ms']:.1f}"
                    d[f"lat@{sz}@{prec}@total"] = f"{x['pre_ms'] + x['inferencia_ms'] + x['post_ms']:.1f}"
                    d[f"lat@{sz}@{prec}@trk"] = f"{x['con_seguimiento_ms']:.1f}"
                    d[f"lat@{sz}@{prec}@fps"] = f"{x['con_seguimiento_fps']:.1f}"
    d.update(cifras_dataset())
    cs = os.path.join(HERE, "runs", "r1", "check_semaforo", "resultado.json")
    if os.path.exists(cs):
        summary["check_semaforo"] = json.load(open(cs, encoding="utf-8"))
        for k, v in summary["check_semaforo"].items():
            mod, et = k.split("|")
            mod = "base" if mod.startswith("base") else "e1old"
            et = "orig" if "original" in et else "arm"
            for m, x in v.items():
                d[f"chk@{mod}@{et}@{m}"] = f"{x:.3f}"
    lines = ["% Generado por yoloe_autolabel/r1_results.py. No editar a mano."]
    for k in sorted(d):
        lines.append(r"\expandafter\def\csname cf@" + k + r"\endcsname{" + d[k] + "}")
    os.makedirs(TAB_DIR, exist_ok=True)
    open(os.path.join(TAB_DIR, "cifras.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    runs = load()
    info = manifest()
    groups_of = {k: v["grupo"] for k, v in info.items()}
    subsets = {
        "principal": [k for k, v in info.items() if v["fuente"] != "SV"],
        "street_view": [k for k, v in info.items() if v["fuente"] == "SV"],
        **{f"fuente_{s}": [k for k, v in info.items() if v["fuente"] == s] for s in ("L", "A", "M", "O")},
    }
    summary = {"modelos_disponibles": {m: len(v) for m, v in runs.items()}, "subconjuntos": {}}
    for sname, names in subsets.items():
        if not names:
            continue
        summary["subconjuntos"][sname] = {m: metrics_on(lst, names) for m, lst in runs.items()}
        summary["subconjuntos"][sname]["n_imagenes"] = len(names)
    main_names = subsets["principal"]
    inst = np.sum([info[k]["c"] for k in main_names], axis=0)
    ngr = [len({info[k]["grupo"] for k in main_names if info[k]["c"][c] > 0}) for c in range(12)]
    summary["instancias_test_principal"] = inst.tolist()
    summary["grupos_test_principal"] = ngr
    summary["clases_estimables"] = [c for c in range(12) if ngr[c] >= 3]
    pairs = [(x, y) for x, y in (("E2", "E1R"), ("E2", "E0"), ("E1R", "E0"), ("E2", "E2noSV"), ("E2", "E0_1024"),
                                  ("E2", "E2_1024"), ("E2c", "E0"), ("E1Rc", "E0")) if x in runs and y in runs]
    summary["bootstrap"] = bootstrap(runs, main_names, groups_of, n_boot=a.n, pairs=pairs)
    # control del recorrido no visto (fuente O): todas las imágenes de su yaml
    runs_o = load(MODELS_SRCO)
    if runs_o:
        names_o = subsets["fuente_O"]
        inst_o = np.sum([info[k]["c"] for k in names_o], axis=0)
        summary["control_recorrido_O"] = {
            "modelos": {m: metrics_on(lst, names_o) for m, lst in runs_o.items()},
            "n_imagenes": len(names_o), "n_grupos": len({info[k]["grupo"] for k in names_o}),
            "instancias": inst_o.tolist()}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    latex_tables(summary)
    cifras_tex(summary)
    json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    p = summary["subconjuntos"]["principal"]
    for m in p:
        if m != "n_imagenes":
            print(m, {k: round(p[m][k]["media"], 4) for k in KEYS})
    print(json.dumps(summary["bootstrap"]["diferencias"], indent=1))


if __name__ == "__main__":
    main()
