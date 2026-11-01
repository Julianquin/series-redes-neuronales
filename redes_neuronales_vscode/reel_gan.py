"""Ep. 6 · GAN · «Estos pueblos no existen».  Uso: python3 reel_gan.py prev f… | chunk 85 | encode"""
import numpy as np, pickle
from PIL import Image, ImageDraw
from reel_common import *
import train_gan_es as T

RUN = pickle.load(open("work/run_gan_es.pkl", "rb")); KEYS = sorted(RUN["snaps"]); MET = RUN["metrics"]; ITERS = RUN["ITERS"]
L = RUN["L"].copy()
for i in range(1, len(L)): L[i] = 0.98 * L[i - 1] + 0.02 * RUN["L"][i]
REAL = RUN["REAL"]; ZF = RUN["Z"]
FAKE_F = T.fwd(RUN["snaps"][KEYS[-1]][0], ZF)[-1]
COLLAPSE = [k for k in KEYS if MET[k][1] < 0.4 and k > 100]
LON0, LON1, LAT0, LAT1 = -9.5, 4.5, 35.85, 43.95; CS = np.cos(np.radians(40))
ASP = (LON1 - LON0) * CS / (LAT1 - LAT0)
def projector(box):
    x0, y0, w = box; h = w / ASP
    def f(xy):
        lon, lat = T.to_lonlat(xy); return x0 + (lon - LON0) * CS / ((LON1 - LON0) * CS) * w, y0 + (LAT1 - lat) / (LAT1 - LAT0) * h
    return f, h
def draw_map(img, box, fake=None, real_col=(84, 96, 110), fake_col=GOLD, real=True, alpha=1.0, frame=True, rr=1.6, rf_=2.6, dmap=None):
    x0, y0, w = box; f, h = projector(box); d = ImageDraw.Draw(img, "RGBA")
    if frame: d.rounded_rectangle((x0 - 10, y0 - 10, x0 + w + 10, y0 + h + 10), radius=22, fill=(14, 20, 30, 255), outline=(44, 58, 74, 255), width=2)
    if dmap is not None: img.alpha_composite(dmap, (int(x0), int(y0))); d = ImageDraw.Draw(img, "RGBA")
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
    if real:
        X, Y = f(REAL)
        for x, y in zip(X, Y): ld.ellipse((x - rr, y - rr, x + rr, y + rr), fill=real_col + (255,))
    if fake is not None:
        X, Y = f(fake)
        for x, y in zip(X, Y):
            if x0 - 4 < x < x0 + w + 4 and y0 - 4 < y < y0 + h + 4: ld.ellipse((x - rf_, y - rf_, x + rf_, y + rf_), fill=fake_col + (230,))
    img.alpha_composite(faded(lay, alpha) if alpha < 1 else lay); return h

def dshade(D, box):
    x0, y0, w = box; h = w / ASP; gw, gh = 140, int(140 / ASP)
    lon = LON0 + (np.arange(gw) + .5) / gw * (LON1 - LON0); lat = LAT1 - (np.arange(gh) + .5) / gh * (LAT1 - LAT0)
    LO, LA = np.meshgrid(lon, lat); xy = T.to_xy(LO.ravel(), LA.ravel())
    p = T.sig(T.fwd(D, xy)[-1][:, 0]).reshape(gh, gw)
    rgba = np.zeros((gh, gw, 4), np.uint8); rgba[..., 0] = np.where(p < .5, 200, 60); rgba[..., 1] = 60; rgba[..., 2] = np.where(p < .5, 70, 200)
    rgba[..., 3] = (np.abs(p - .5) * 2 * 70).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").resize((int(w), int(h)), Image.BILINEAR)

def s_hook(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, [f"Estos {fmt(len(ZF))} pueblos", "no existen"], 320, size=84)
    h = draw_map(img, (100, 470, 880), fake=FAKE_F, real_col=(46, 56, 68), alpha=1.0)
    d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Los ha colocado una red neuronal", "que nunca ha visto un mapa"], 1210, size=42)
    small(d, "en gris, los 8.042 municipios reales de la península y Baleares", 1350, col=MUTED, size=26)
    return img

def s_duel(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Falsificador", "contra detective"], 320, size=84)
    for k, (x, title, sub, col) in enumerate(((90, "GENERADOR", "coloca pueblos falsos", GOLD), (560, "DISCRIMINADOR", "¿real o falso?", BLUE))):
        a = ease((rf - k * 12) / 10)
        if a <= 0: continue
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rounded_rectangle((x, 480, x + 430, 680), radius=24, fill=(18, 26, 36, 255), outline=col + (255,), width=5)
        ld.text((x + 215, 545), title, font=font(FS["head"], 40), fill=col, anchor="mm"); ld.text((x + 215, 615), sub, font=font(FS["body"], 34), fill=SOFT, anchor="mm")
        img.alpha_composite(faded(lay, a))
    d = ImageDraw.Draw(img, "RGBA")
    if rf > 26:
        a = ease((rf - 26) / 12); lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_map(lay, (100, 760, 880), fake=None, real_col=(225, 230, 236)); img.alpha_composite(faded(lay, a)); d = ImageDraw.Draw(img, "RGBA")
        small(d, "lo único que ve el detective: dónde hay pueblos de verdad", 1450, col=SOFT, size=28)
    return img

NTR = 180
W8 = np.array([3.0 if k in COLLAPSE else 1.0 for k in KEYS]); CUM = np.cumsum(W8) / W8.sum()
def kidx(rf): return int(np.searchsorted(CUM, min(1.0, rf / (NTR - 24)) * 0.9999))
DCACHE = {}
def s_train(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["Aprenden", "peleando"], 290, size=84)
    k = KEYS[kidx(rf)]; G, D = RUN["snaps"][k]; F = T.fwd(G, ZF)[-1]
    if k not in DCACHE: DCACHE.clear(); DCACHE[k] = dshade(D, (100, 450, 880))
    draw_map(img, (100, 450, 880), fake=F, dmap=DCACHE[k]); d = ImageDraw.Draw(img, "RGBA")
    chip(d, 120, 470, f"iteración {fmt(k)}", col=WHITE, size=26)
    land = MET[k][0]; coll = k in COLLAPSE
    if coll: chip(d, W / 2, 1060, "¡SE ATASCA EN UNA ZONA! (mode collapse)", col=RED, size=30, anchor="mm", bg=(40, 14, 18, 240))
    d.text((W / 2, 1150), T_(f"{dec(land * 100, 0)} % en tierra firme"), font=font(FS["head"], 64), fill=GOLD if land > .83 and not coll else WHITE, anchor="mm")
    xs = np.arange(1, len(L) + 1)[::10]
    chart(img, (110, 1200, 970, 1470), [dict(xs=xs, ys=L[::10, 1], upto=k, col=GOLD, name="falsificador", val_fmt=lambda v: dec(v, 2)),
                                        dict(xs=xs, ys=L[::10, 0], upto=k, col=BLUE, name="detective", val_fmt=lambda v: dec(v, 2))],
          "pérdida de cada red", xmax=ITERS, ylim=(0.5, 1.4), yticks=[0.6, 0.9, 1.2], logy=False, yfmt=lambda v: dec(v, 1))
    return img

rs = np.random.default_rng(3); REAL_S = REAL[rs.choice(len(REAL), len(ZF), replace=False)]
CH_A, CH_B = FAKE_F, REAL_S                              # A = falso, B = real
def s_challenge(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué mapa es", "el de verdad?"], 300, size=82)
    rev = rf >= 84
    for i, (pts, lab, ok) in enumerate(((CH_A, "A", False), (CH_B, "B", True))):
        y = 440 + i * 500; col = (GREEN if ok else GOLD) if rev else WHITE
        h = draw_map(img, (190, y, 620), fake=pts, real=False, fake_col=(236, 236, 228), rf_=1.8); d = ImageDraw.Draw(img, "RGBA")
        d.ellipse((92, y + h / 2 - 40, 172, y + h / 2 + 40), fill=(22, 30, 40), outline=col, width=4)
        d.text((132, y + h / 2), lab, font=font(FS["head"], 46), fill=col, anchor="mm")
        if rev: chip(d, 190 + 620 - 12, y + 12, "REAL" if ok else "FALSO · lo hizo la red", col=col, size=28, anchor="rt", bg=(8, 12, 18, 235))
    if not rev and rf >= 20:
        k = min(int((rf - 20) // 20), 2); p = ((rf - 20) % 20) / 20 if rf < 80 else 1
        d.ellipse((W / 2 - 50, 1380, W / 2 + 50, 1480), outline=(40, 52, 66), width=9)
        d.arc((W / 2 - 50, 1380, W / 2 + 50, 1480), start=-90, end=-90 + 360 * p, fill=GOLD, width=9)
        d.text((W / 2, 1430), str(3 - k), font=font(FS["head"], 52), fill=WHITE, anchor="mm")
    if rev and rf > 96:
        body(d, ["Pista: el falso se sale al mar y a Portugal"], 1440, size=36, col=GOLD)
    return img

def s_end(rf, n):
    img = BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    headline(d, ["¿Qué país", "le pongo?"], 330)
    draw_map(img, (190, 500, 700), fake=FAKE_F, real_col=(46, 56, 68)); d = ImageDraw.Draw(img, "RGBA")
    body(d, ["Comenta un país y le dibujo sus pueblos falsos"], 1090, size=40)
    d.text((W / 2, 1180), T_("Redes Neuronales · Ep. 6"), font=font(FS["monob"], 32), fill=GOLD, anchor="mm")
    d.text((W / 2, 1270), HANDLE, font=font(FS["head"], 56), fill=WHITE, anchor="mm")
    return img

TL = Timeline([("hook", 75), ("duel", 90), ("train", NTR), ("challenge", 120), ("end", 75)],
              dict(hook=s_hook, duel=s_duel, train=s_train, challenge=s_challenge, end=s_end))

def make_audio():
    A = Audio(TL.N); A.pad((110, 164.8, 220, 261.6))
    for nm in ("duel", "train", "challenge", "end"): A.at(TL.start(nm) - 6, A.whoosh(), 0.2)
    A.at(0, A.tone(55, 0.6, 9), 0.5)
    A.at(TL.start("duel"), A.tone(392, 0.5, 6), 0.12); A.at(TL.start("duel") + 12, A.tone(330, 0.5, 6), 0.12)
    t0 = TL.start("train"); was = False
    for rf in range(NTR):
        c = KEYS[kidx(rf)] in COLLAPSE
        if c and not was: A.at(t0 + rf, A.tone(140, 0.9, 3) + A.tone(147, 0.9, 3), 0.2)
        was = c
    tick = A.tone(880, 0.18, 30)
    for k in range(3): A.at(TL.start("challenge") + 20 + 20 * k, tick, 0.15)
    A.at(TL.start("challenge") + 84, A.tone(660, 0.8, 5) + 0.6 * A.tone(990, 0.8, 5), 0.15)
    A.chime(TL.start("end")); return A

if __name__ == "__main__":
    cli(TL, "ep6_reel_gan_espana", make_audio)
