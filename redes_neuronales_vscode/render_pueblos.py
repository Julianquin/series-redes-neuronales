import numpy as np, pickle, json, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R

W, H, FPS = 1080, 1920, 30
P1, RD = 150, 150
TRAIN_F = P1 + 2 * RD            # 450
HOLD_F, END_F = 20, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
GREEN = (96, 206, 140); INK = (14, 16, 20); PAPER = (244, 244, 238)
HANDLE = R.HANDLE
run = pickle.load(open("work/run_pueblos.pkl", "rb"))
SEL = json.load(open("data/pueblos_sel.json"))
L, CK = run["L"], run["CK"]; ITERS = CK[-1]
Ls = np.convolve(L, np.ones(25) / 25, "same"); Ls[:12] = L[:12]
EVO = SEL["evo"]; ROUNDS = SEL["rounds"]
fmt = lambda n: f"{n:,}".replace(",", ".")

_fc = {}
def sg(size, w=700):
    if (size, w) not in _fc: _fc[(size, w)] = R.font(R.SG, size, w)
    return _fc[(size, w)]
fH = R.font(R.JB, 24, 500); fS = R.font(R.JB, 22, 400); fCk = R.font(R.JB, 24, 600)
ease = lambda t: 0 if t <= 0 else 1 if t >= 1 else t * t * (3 - 2 * t)
clamp = lambda x, a=0, b=1: max(a, min(b, x))

def check(d_, x, y, s, col, w=6):
    d_.line([(x - s * .5, y), (x - s * .12, y + s * .42), (x + s * .55, y - s * .42)], fill=col, width=w, joint="curve")
def cross(d_, x, y, s, col, w=6):
    d_.line([(x - s * .42, y - s * .42), (x + s * .42, y + s * .42)], fill=col, width=w); d_.line([(x - s * .42, y + s * .42), (x + s * .42, y - s * .42)], fill=col, width=w)

def header(d_):
    d_.text((W / 2, 112), f"REDES NEURONALES · EP. 3 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d_.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "inventarse un pueblo"
    tb = d_.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d_.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d_.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

SS = 2
def sign_layer(w, h, full, shown, edge=None, glow=None, caret=False):
    pad = 50; lay = Image.new("RGBA", ((w + 2 * pad) * SS, (h + 2 * pad) * SS), (0, 0, 0, 0))
    if glow:
        gl = Image.new("RGBA", lay.size, (0, 0, 0, 0)); ImageDraw.Draw(gl).rounded_rectangle((pad * SS, pad * SS, (pad + w) * SS, (pad + h) * SS), radius=26 * SS, fill=glow + (190,))
        lay.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18 * SS)))
    d_ = ImageDraw.Draw(lay)
    box = (pad * SS, pad * SS, (pad + w) * SS, (pad + h) * SS)
    d_.rounded_rectangle(box, radius=24 * SS, fill=PAPER, outline=edge or (30, 34, 40), width=(5 if edge else 3) * SS)
    ins = 13 * SS; d_.rounded_rectangle((box[0] + ins, box[1] + ins, box[2] - ins, box[3] - ins), radius=14 * SS, outline=INK, width=6 * SS)
    size = 110
    while size > 30 and sg(size).getlength(full) > (w - 100): size -= 4
    fs = ImageFont.truetype(R.SG, size * SS)
    try: fs.set_variation_by_axes([700])
    except Exception: pass
    tw = fs.getlength(full); x0 = box[0] + (box[2] - box[0] - tw) / 2; cy = (box[1] + box[3]) / 2 + 2 * SS
    d_.text((x0, cy), shown, font=fs, fill=INK, anchor="lm")
    if caret:
        cx = x0 + fs.getlength(shown) + 6 * SS; d_.rectangle((cx, cy - size * SS * .36, cx + 7 * SS, cy + size * SS * .36), fill=GOLD)
    return lay.resize((lay.width // SS, lay.height // SS), Image.LANCZOS)

def paste_c(img, lay, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1: lay = lay.resize((int(lay.width * scale), int(lay.height * scale)), Image.LANCZOS)
    if alpha < 1: lay = lay.copy(); lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
    img.alpha_composite(lay, (int(cx - lay.width / 2), int(cy - lay.height / 2)))

# ---------------- loss panel ----------------
LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
ymin, ymax = 1.2, 3.7
def lpx(it, v): return LX0 + (it - 1) / (len(L) - 1) * (LX1 - LX0), LY1 - (clamp((v - ymin) / (ymax - ymin))) * (LY1 - LY0)
def loss_panel(d_, upto, ck_i):
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((90, 1478), "error de la red por letra", font=R.fMono, fill=GREY, anchor="lm")
    d_.line((800, 1478, 830, 1478), fill=BLUE, width=4); d_.text((842, 1478), f"{Ls[upto - 1]:.3f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v in (1.5, 2.0, 2.5, 3.0, 3.5):
        _, yy = lpx(1, v); d_.line((LX0, yy, LX1, yy), fill=(26, 36, 46), width=1); d_.text((LX0 - 14, yy), f"{v:.1f}", font=R.fMonoS, fill=GREY, anchor="rm")
    d_.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 1000, 2000, 3000):
        x = LX0 + e / ITERS * (LX1 - LX0); d_.text((x, LY1 + 26), fmt(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d_.text(((LX0 + LX1) / 2, LY1 + 56), "iteración", font=R.fMonoS, fill=GREY, anchor="mm")
    pts = [lpx(i, Ls[i - 1]) for i in range(1, upto + 1, max(1, upto // 500))] + [lpx(upto, Ls[upto - 1])]
    if len(pts) > 1: d_.line(pts, fill=BLUE, width=3, joint="curve")
    for j, ck in enumerate(CK[:ck_i + 1]):
        if ck <= upto:
            x, y = lpx(ck, Ls[ck - 1]); d_.ellipse((x - 8, y - 8, x + 8, y + 8), fill=GOLD if j == ck_i else GREY, outline=(8, 12, 16), width=2)

_pool = sorted(x['n'] for x in run['recs'] if x['c'] == SEL['comm1'] and 8 <= len(x['n']) <= 11 and ' ' not in x['n'])
REFS = [_pool[i] for i in np.random.default_rng(3).choice(len(_pool), 3, replace=False)]
CAPS = ["ruido puro", "empieza a separar palabras", "sílabas casi reales", "ya suena a pueblo", "un pueblo que no existe"]

def part1(f):
    ci = min(4, f // 30); lf = f - ci * 30; prev = CK[ci - 1] if ci else 1
    upto = int(prev + (CK[ci] - prev) * ease(lf / 12)); full = EVO[ci]
    shown_n = int(clamp((lf - 3) * len(full) / 16, 0, len(full)))
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"iter {fmt(CK[ci])} / {fmt(ITERS)}", font=R.fMono, fill=GREY, anchor="lm")
    d_.text((988, 425), "LSTM(128) · condicionada por comunidad", font=R.fMono, fill=GREY, anchor="rm")
    for j, c in enumerate(CK):
        x = 100 + j * 180; on = j == ci; done = j < ci
        d_.rounded_rectangle((x, 462, x + 160, 504), radius=21, fill=(40, 32, 14) if on else (18, 25, 33), outline=GOLD if on else (EDGE if not done else (80, 96, 112)), width=2)
        d_.text((x + 80, 483), f"iter {fmt(c)}", font=fCk, fill=GOLD if on else (GREY if done else (70, 84, 98)), anchor="mm")
    d_.text((540, 590), "comunidad", font=fS, fill=GREY, anchor="mm")
    cw = sg(46).getlength(SEL["comm1"]) + 70
    d_.rounded_rectangle((540 - cw / 2, 615, 540 + cw / 2, 685), radius=35, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d_.text((540, 650), SEL["comm1"], font=sg(46), fill=WHITE, anchor="mm")
    done = ci == 4 and shown_n >= len(full)
    lay = sign_layer(860, 250, full, full[:shown_n], caret=shown_n < len(full) or (f // 8) % 2 == 0, glow=GOLD if done else None, edge=GOLD if done else None)
    paste_c(img, lay, 540, 880); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((540, 1050), CAPS[ci], font=sg(34, 600), fill=GOLD if ci == 4 else GREY, anchor="mm")
    if not done:
        d_.text((540, 1140), "así suenan los que sí existen", font=fS, fill=GREY, anchor="mm")
        xs = [210, 540, 870]
        for nm_, x_ in zip(REFS, xs):
            cw_ = sg(34, 600).getlength(nm_) + 50
            d_.rounded_rectangle((x_ - cw_ / 2, 1180, x_ + cw_ / 2, 1244), radius=14, fill=(18, 25, 33), outline=EDGE, width=2)
            d_.text((x_, 1212), nm_, font=sg(34, 600), fill=WHITE, anchor="mm")
    if done:
        a = ease((lf - 22) / 6)
        if a > 0:
            lay2 = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay2)
            ld.rounded_rectangle((150, 1110, 930, 1300), radius=16, fill=(16, 24, 32, 255), outline=GREEN + (255,), width=2)
            check(ld, 215, 1160, 34, GREEN, 7)
            ld.text((255, 1160), "NO EXISTE · lo ha inventado la red", font=R.fMono, fill=GREEN, anchor="lm")
            ld.text((185, 1220), "lo más parecido que existe:", font=fS, fill=GREY, anchor="lm")
            ld.text((185, 1262), f"{SEL['near_name']} ({SEL['near_prov']})", font=sg(38, 700), fill=WHITE, anchor="lm")
            lay2.putalpha(lay2.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay2)
    d_ = ImageDraw.Draw(img, "RGBA"); loss_panel(d_, upto, ci)
    return img

def ring(d_, cx, cy, r, p, num):
    d_.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(34, 46, 58), width=10)
    if p > 0: d_.arc((cx - r, cy - r, cx + r, cy + r), start=-90, end=-90 + 360 * p, fill=GOLD, width=10)
    d_.text((cx, cy - 2), str(num), font=sg(78), fill=WHITE, anchor="mm")

SLOT_Y = [640, 870, 1100]; LETTERS = "ABC"
def reveal_times(items):
    real_idx = [i for i, it in enumerate(items) if it["real"]]; rev = {}
    for k, i in enumerate(real_idx): rev[i] = 86 + 12 * k
    inv_i = next(i for i, it in enumerate(items) if not it["real"]); rev[inv_i] = 86 + 12 * len(real_idx) + 8
    return rev, inv_i

def round_frame(rd, rf):
    R_ = ROUNDS[rd]
    img = R.BGIMG.copy(); d_ = ImageDraw.Draw(img, "RGBA"); header(d_)
    R.panel(d_, (60, 380, 1020, 1410))
    d_.text((92, 425), f"RONDA {rd + 1} / 2", font=R.fMono, fill=GOLD, anchor="lm")
    d_.text((988, 425), f"comunidad: {R_['comm']}", font=R.fMono, fill=WHITE, anchor="rm")
    d_.text((540, 500), "Dos pueblos existen. Uno lo ha inventado la red.", font=sg(32, 600), fill=WHITE, anchor="mm")
    items = R_["items"]; rev, inv_i = reveal_times(items)
    revealed_inv = rf >= rev[inv_i]
    for i, it in enumerate(items):
        cy = SLOT_Y[i]; ap = ease((rf - 2 - i * 5) / 12)
        if ap <= 0: continue
        t_rev = rf - rev[i]; is_rev = t_rev >= 0
        glow = (GREEN if it["real"] else GOLD) if is_rev else None
        lay = sign_layer(740, 150, it["name"], it["name"], edge=glow, glow=glow)
        sc = 1.0
        if is_rev and not it["real"]: sc = 1 + 0.07 * np.exp(-t_rev / 5) * np.cos(t_rev * 0.9)
        paste_c(img, lay, 600, cy + (1 - ap) * 40, scale=sc, alpha=ap); d_ = ImageDraw.Draw(img, "RGBA")
        d_.ellipse((120, cy - 42, 204, cy + 42), fill=(22, 30, 40), outline=glow or (62, 78, 96), width=3)
        d_.text((162, cy - 2), LETTERS[i], font=sg(52), fill=glow or WHITE, anchor="mm")
        if is_rev:
            a = ease(t_rev / 6); lay2 = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay2)
            if it["real"]:
                check(ld, 262, cy + 98, 26, GREEN, 6); ld.text((290, cy + 98), f"REAL · {it['prov']}", font=R.fMono, fill=GREEN, anchor="lm")
            else:
                cross(ld, 262, cy + 98, 26, GOLD, 6); ld.text((290, cy + 98), "INVENTADO POR LA RED", font=R.fMono, fill=GOLD, anchor="lm")
            lay2.putalpha(lay2.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay2); d_ = ImageDraw.Draw(img, "RGBA")
    if rf < 86:
        if rf >= 22:
            k = min(int((rf - 22) // 20), 2); p = ((rf - 22) % 20) / 20 if rf < 82 else 1
            ring(d_, 540, 1295, 62, p, 3 - k)
            d_.text((540, 1378), "¿cuál es el inventado?", font=sg(32, 600), fill=GREY, anchor="mm")
        else:
            d_.text((540, 1330), "¿cuál es el inventado?", font=sg(36, 600), fill=GREY, anchor="mm")
    elif revealed_inv:
        a = ease((rf - rev[inv_i] - 6) / 8)
        if a > 0:
            lay2 = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay2); inv = items[inv_i]
            ld.text((540, 1290), f"{LETTERS[inv_i]} era el falso", font=sg(46), fill=GOLD, anchor="mm")
            ld.text((540, 1348), f"se parece a {inv['near']} ({inv['prov']}), que sí existe", font=fS, fill=GREY, anchor="mm")
            lay2.putalpha(lay2.getchannel("A").point(lambda v: int(v * a))); img.alpha_composite(lay2); d_ = ImageDraw.Draw(img, "RGBA")
    R.panel(d_, (60, 1440, 1020, 1840))
    d_.text((W / 2, 1520), "¿lo adivinaste?", font=sg(72), fill=WHITE, anchor="mm")
    d_.text((W / 2, 1604), "Comenta tu comunidad", font=sg(40, 600), fill=GOLD, anchor="mm")
    d_.text((W / 2, 1654), "y te invento un pueblo", font=sg(40, 600), fill=GOLD, anchor="mm")
    d_.line((120, 1706, 960, 1706), fill=(26, 36, 46), width=1)
    d_.text((W / 2, 1752), f"{fmt(SEL['nnames'])} pueblos vistos · {SEL['ncomm']} comunidades", font=fS, fill=GREY, anchor="mm")
    d_.text((W / 2, 1788), f"{fmt(SEL['nparams'])} parámetros · {fmt(ITERS)} iteraciones", font=fS, fill=GREY, anchor="mm")
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy()
    lay = sign_layer(760, 200, "¿Dónde estoy?", "¿Dónde estoy?", edge=GOLD, glow=GOLD); paste_c(img, lay, 540, 700)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d_ = ImageDraw.Draw(img, "RGBA")
    d_.text((W / 2, 1030), "Ep. 3 · Pueblos que no existen", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d_, (200, 1120, 880, 1380))
    d_.text((W / 2, 1180), "COMENTA TU COMUNIDAD", font=fFollow, fill=GOLD, anchor="mm")
    d_.text((W / 2, 1250), "y te invento un pueblo", font=R.fMono, fill=GREY, anchor="mm")
    d_.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

def audio():
    sr = 44100; n = int(N / FPS * sr); t = np.arange(n) / sr; a = np.zeros(n)
    env = np.clip(t / 1.2, 0, 1) * np.clip((N / FPS - t) / 1.0, 0, 1)
    for fr in (98, 146.8, 196, 246.9): a += 0.04 * env * np.sin(2 * np.pi * fr * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    rng = np.random.default_rng(0); cl = rng.normal(0, 1, int(0.012 * sr)) * np.exp(-np.arange(int(0.012 * sr)) / (0.002 * sr))
    def at(fr_, arr, g):
        i0 = int(fr_ / FPS * sr); m = min(len(arr), n - i0)
        if m > 0: a[i0:i0 + m] += g * arr[:m]
    for ci in range(5):
        L_ = len(EVO[ci])
        for k in range(L_): at(ci * 30 + 3 + k * 16 / max(L_, 1), cl, 0.12)
        tt = np.arange(int(.5 * sr)) / sr; at(ci * 30, np.exp(-tt * 6) * np.sin(2 * np.pi * (440 + 110 * ci) * tt), 0.10)
    tick = np.exp(-np.arange(int(.18 * sr)) / (0.03 * sr)) * np.sin(2 * np.pi * 880 * np.arange(int(.18 * sr)) / sr)
    for rd in range(2):
        s0 = P1 + rd * RD
        for k in range(3): at(s0 + 22 + 20 * k, tick, 0.18)
        rev, inv_i = reveal_times(ROUNDS[rd]["items"])
        for i, fr_ in rev.items():
            tt = np.arange(int(.7 * sr)) / sr; isinv = i == inv_i; fr0 = 523.3 if isinv else 660
            snd = np.exp(-tt * 5) * (np.sin(2 * np.pi * fr0 * tt) + (0.8 * np.sin(2 * np.pi * fr0 * 1.26 * tt) if isinv else 0.6 * np.sin(2 * np.pi * fr0 * 1.5 * tt)))
            at(s0 + fr_, snd, 0.18 if isinv else 0.12)
    i0 = int((TRAIN_F + HOLD_F) / FPS * sr); tt = np.arange(n - i0) / sr
    for fr in (880, 1318.5, 1760): a[i0:] += 0.08 * np.exp(-tt * 2.5) * np.sin(2 * np.pi * fr * tt)
    a = a / np.abs(a).max() * 0.8
    with wave.open("work/pueblos.wav", "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes((a * 32767).astype(np.int16).tobytes())

def scene(f):
    if f < P1: return part1(f)
    if f < P1 + RD: return round_frame(0, f - P1)
    return round_frame(1, min(f - P1 - RD, RD - 1))

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = round_frame(1, RD - 1)
    def frame(f):
        if f < TRAIN_F: return scene(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/p{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/pueblos.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait(); audio()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/pueblos.mp4", "-i", "work/pueblos.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep3b_pueblos_inventados.mp4"])
