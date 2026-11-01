"""Ep. 7 · GNN · «¿De qué comunidad es este pueblo?»  Uso: python3 reel_gnn.py prev f… | chunk 85 | encode"""
import numpy as np, pickle
from PIL import Image, ImageDraw
from reel_common import *

R = pickle.load(open("work/run_gnn_es.pkl", "rb"))
NAMES, PROV, COMM, Y, LAT, LON, TR, TE, KEYS = R["names"], R["prov"], R["COMM"], R["y"], R["lat"], R["lon"], R["TR"], R["TE"], R["keys"]
EDGES = R["EDGES"]; N = len(NAMES); TEset = np.zeros(N, bool); TEset[TE] = True
PAL = [(230, 90, 90), (240, 160, 70), (250, 214, 90), (170, 220, 90), (90, 200, 120), (70, 200, 190), (80, 170, 240), (110, 120, 240), (170, 110, 235),
       (230, 110, 200), (200, 150, 120), (150, 190, 160), (240, 130, 140), (120, 220, 230), (220, 200, 150), (160, 160, 175), (255, 240, 200)]
CCOL = {c: PAL[i % len(PAL)] for i, c in enumerate(COMM)}
FINAL = {k: R[k]["preds"][KEYS[-1]] for k in ("mlp", "gnn")}
ACC = {k: R[k]["L"][:, 2] for k in ("mlp", "gnn")}
CAN = np.array([c == "Canarias" for c in (COMM[i] for i in Y)])
LON0, LON1, LAT0, LAT1 = -9.5, 4.5, 35.85, 43.95; CS = np.cos(np.radians(40)); ASP = (LON1 - LON0) * CS / (LAT1 - LAT0)

def proj(box):
    x0, y0, w = box; h = w / ASP; px = np.zeros(N); py = np.zeros(N); m = ~CAN
    px[m] = x0 + (LON[m] - LON0) / (LON1 - LON0) * w; py[m] = y0 + (LAT1 - LAT[m]) / (LAT1 - LAT0) * h
    ix, iy, iw = x0 + 4, y0 + h * 0.80, w * 0.30; ih = iw * (2.2 / ((18.3 - 13.3) * np.cos(np.radians(28.5))))      # recuadro de Canarias
    px[CAN] = ix + (LON[CAN] + 18.3) / 5.0 * iw; py[CAN] = iy + (29.5 - LAT[CAN]) / 2.2 * ih * 0.95
    return px, py, h, (ix, iy, iw, ih)
def draw_map(img, box, cols, mask=None, r=2.2, frame=True):
    x0, y0, w = box; px, py, h, (ix, iy, iw, ih) = proj(box); d = ImageDraw.Draw(img, "RGBA")
    if frame: d.rounded_rectangle((x0 - 10, y0 - 10, x0 + w + 10, y0 + h + 10), radius=22, fill=(14, 20, 30, 255), outline=(44, 58, 74, 255), width=2)
    d.rectangle((ix - 4, iy - 4, ix + iw, iy + ih), outline=(70, 86, 104), width=2)
    for i in range(N):
        if mask is not None and not mask[i]: continue
        c = cols[i]; d.ellipse((px[i] - r, py[i] - r, px[i] + r, py[i] + r), fill=c)
    return px, py, h

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Una red ha dibujado", "las comunidades"], 320, size=80)
    p = FINAL["gnn"]; cols = [CCOL[COMM[Y[i] if not TEset[i] else p[i]]] for i in range(N)]
    a = ease(rf / 10); lay = BGIMG.copy(); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_map(lay, (100, 470, 880), cols, r=2.6); img.alpha_composite(faded(lay, a)); d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Sin ver ni una coordenada: solo nombres", "de pueblos y quién es vecino de quién"], 1210, size=40)
    small(d, f"{fmt(N)} municipios · cada color, la comunidad que la red cree", 1350, col=MUTED, size=26)
    return img

# ejemplo: nombre genérico, la red de nombre falla y la de vecinos acierta
def pick_example():
    nb = [[] for _ in range(N)]
    for a, b in EDGES: nb[a].append(b); nb[b].append(a)
    best = None
    for i in TE:
        if FINAL["mlp"][i] != Y[i] and FINAL["gnn"][i] == Y[i] and COMM[Y[i]] in ("Cataluña", "Galicia", "País Vasco", "Illes Balears") and len(NAMES[i]) <= 12:
            nn = [j for j in nb[i] if len(NAMES[j]) <= 16][:6]
            if len(nn) >= 6: best = (i, nn); break
    return best
EX, EXN = pick_example()
def s_task(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿De qué comunidad", "es este pueblo?"], 320, size=80)
    cx, cy = W / 2, 860; a1 = ease(rf / 8)
    d.rounded_rectangle((cx - 230, cy - 60, cx + 230, cy + 60), radius=30, fill=(244, 244, 238, int(255 * a1)), outline=GOLD + (int(255 * a1),), width=5)
    d.text((cx, cy), NAMES[EX], font=font(FS["monob"], 52), fill=(14, 16, 20, int(255 * a1)), anchor="mm")
    small(d, "pista 1: su nombre", cy + 100, col=SOFT, size=30)
    if rf > 26:
        a2 = ease((rf - 26) / 14); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        for k, j in enumerate(EXN):
            ang = -np.pi / 2 + k * 2 * np.pi / len(EXN) + 0.3; x = cx + 380 * np.cos(ang); y = cy + 290 * np.sin(ang)
            ld.line((cx, cy, x, y), fill=(120, 136, 156, 255), width=3)
            f = font(FS["mono"], 30); tw = f.getlength(NAMES[j])
            ld.rounded_rectangle((x - tw / 2 - 16, y - 28, x + tw / 2 + 16, y + 28), radius=28, fill=(30, 40, 54, 255))
            ld.text((x, y), NAMES[j], font=f, fill=WHITE, anchor="mm")
        img.alpha_composite(faded(lay, a2)); d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle((cx - 230, cy - 60, cx + 230, cy + 60), radius=30, fill=(244, 244, 238), outline=GOLD, width=5)
        d.text((cx, cy), NAMES[EX], font=font(FS["monob"], 52), fill=(14, 16, 20), anchor="mm")
        small(d, "pista 2: los nombres de sus vecinos", 1240, col=GOLD, size=32)
    if rf > 70:
        chip(d, W / 2, 1320, f"es de {COMM[Y[EX]]} ({PROV[EX]})", col=GREEN, size=34, anchor="mm", bg=(14, 30, 22, 255))
    return img

NTR = 165
def kidx(rf): return int(round(min(1.0, rf / (NTR - 25)) * (len(KEYS) - 1)))
def s_race(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Dos redes,", "dos pistas"], 290, size=82)
    e = KEYS[kidx(rf)]
    for k, (kind, lab, col, y) in enumerate((("mlp", "solo el nombre", WHITE, 445), ("gnn", "nombre + vecinos", GOLD, 950))):
        p = R[kind]["preds"][e]; cols = [GREEN if p[i] == Y[i] else RED for i in range(N)]
        draw_map(img, (230, y, 620), cols, mask=TEset, r=1.9); d = ImageDraw.Draw(img, "RGBA")
        chip(d, 230, y - 2, T_(lab), col=col, size=28)
        chip(d, 860, y + 30, f"{dec(ACC[kind][e - 1] * 100)} %", col=col, size=40, anchor="rt", bg=(8, 12, 18, 235))
    chip(d, W / 2, 925, f"época {e} · verde acierta, rojo falla", col=SOFT, size=24, anchor="mm", bold=False, bg=(30, 40, 54, 255))
    return img

def s_curve(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["La diferencia,", "en una curva"], 320, size=84)
    E = len(ACC["gnn"]); up = 10 ** (np.log10(E) * ease((rf - 4) / 50)) if rf > 4 else 0; xs = np.arange(1, E + 1)
    chart(img, (90, 470, 990, 960), [dict(xs=xs, ys=R["mlp"]["L"][:, 1], upto=up, col=WHITE, name="nombre", val_fmt=lambda v: dec(v, 2)),
                                     dict(xs=xs, ys=R["gnn"]["L"][:, 1], upto=up, col=GOLD, name="+ vecinos", val_fmt=lambda v: dec(v, 2))],
          "error en el examen", xmax=E, ylim=(0.2, 4.5), yticks=[0.3, 1, 3], xlabel="época", yfmt=lambda v: dec(v, 1) if v < 1 else str(int(v)))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 50:
        a = ease((rf - 50) / 10); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        for x, lab, v, col in ((300, "solo el nombre", ACC["mlp"][-1], WHITE), (780, "nombre + vecinos", ACC["gnn"][-1], GOLD)):
            ld.text((x, 1060), f"{dec(v * 100)} %", font=font(FS["head"], 76), fill=col, anchor="mm")
            ld.text((x, 1130), lab, font=font(FS["mono"], 30), fill=SOFT, anchor="mm")
        body(ld, ["La red del nombre memoriza: su error de examen sube.", "Decir siempre «Castilla y León» ya da un 28 %."], 1260, size=34)
        img.alpha_composite(faded(lay, a))
    return img

ENCL = [("Condado de Treviño", "enclave de Burgos en Álava"), ("Ademuz", "Rincón de Ademuz (Valencia)")]
def s_errors(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Dónde", "se equivoca?"], 320, size=84)
    p = FINAL["gnn"]; err = TEset & (p != Y); cols = [RED if err[i] else (52, 64, 78) for i in range(N)]
    px, py, h = draw_map(img, (100, 480, 880), cols, r=2.4); d = ImageDraw.Draw(img, "RGBA")
    for k, (nm, note) in enumerate(ENCL):
        a = ease((rf - 20 - k * 22) / 10)
        if a <= 0: continue
        i = NAMES.index(nm); x, y = px[i], py[i]; ty = 1210 + k * 110
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.ellipse((x - 16, y - 16, x + 16, y + 16), outline=GOLD, width=5); ld.line((x, y + 16, 200 + k * 360, ty - 40), fill=GOLD, width=3)
        ld.text((120 + k * 440, ty - 18), nm, font=font(FS["monob"], 30), fill=GOLD, anchor="lm")
        ld.text((120 + k * 440, ty + 22), f"dice {COMM[p[i]]}", font=font(FS["mono"], 26), fill=SOFT, anchor="lm")
        ld.text((120 + k * 440, ty + 56), note, font=font(FS["mono"], 22), fill=MUTED, anchor="lm")
        img.alpha_composite(faded(lay, a))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 66: body(d, ["Falla en las fronteras y en los enclaves:", "sus vecinos son de otra comunidad"], 1420, size=34, col=WHITE)
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Ha acertado", "con tu pueblo?"], 330)
    p = FINAL["gnn"]; cols = [CCOL[COMM[Y[i] if not TEset[i] else p[i]]] for i in range(N)]
    draw_map(img, (190, 500, 700), cols, r=2.1); d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Comenta tu pueblo y te digo qué piensa la red"], 1090, size=40)
    d.text((W / 2, 1180), T_("Redes Neuronales · Ep. 7"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1270), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 75), ("task", 90), ("race", NTR), ("curve", 90), ("errors", 105), ("end", 75)],
              dict(hook=s_hook, task=s_task, race=s_race, curve=s_curve, errors=s_errors, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad((110, 164.8, 220, 246.9))
    for nm in ("task", "race", "curve", "errors", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.2)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    t0 = TL.start("task")
    for k in range(len(EXN)): A.at(t0 + 26 + k * 2, A.tone(660 + 40 * k, 0.25, 14), 0.07)
    A.at(t0 + 70, A.tone(660, 0.8, 5) + 0.6 * A.tone(990, 0.8, 5), 0.14)
    r0 = TL.start("race"); seg = np.arange(int(NTR / FPS * A.sr)) / A.sr
    acc = np.interp(seg * FPS, np.arange(NTR), [ACC["gnn"][KEYS[kidx(rf)] - 1] for rf in range(NTR)])
    A.at(r0, 0.05 * np.sin(2 * np.pi * np.cumsum(220 * 2 ** (acc * 1.5)) / A.sr), 1.0)
    for k in range(2): A.at(TL.start("errors") + 20 + 22 * k, A.tone(196, 0.5, 6), 0.14)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep7_reel_gnn_comunidades", make_audio)
