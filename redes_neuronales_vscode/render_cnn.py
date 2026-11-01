import numpy as np, pickle, subprocess, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import render_mlp as R   # reuse palette, fonts, background, panel(), end-card glow style

W, H, FPS = 1080, 1920, 30
TRAIN_F, HOLD_F, END_F = 390, 24, 66
N = TRAIN_F + HOLD_F + END_F
GOLD, BLUE, WHITE, GREY, EDGE, PANEL = R.GOLD, R.BLUE, R.WHITE, R.GREY, R.EDGE, R.PANEL
GREEN, REDC = (96, 206, 140), (236, 96, 96)
HANDLE = R.HANDLE
run = pickle.load(open("work/run_cnn.pkl", "rb"))
Xva, yva, Cva, L, snaps = run["Xva"], run["yva"], run["Cva"], run["L"], run["snaps"]
keys = sorted(snaps); E = len(L)

def fwd(P, C):
    Z = C @ P["Wc"] + P["bc"]; A = np.maximum(Z, 0).reshape(-1, 8, 8, 8)
    F = A.reshape(-1, 4, 2, 4, 2, 8).max((2, 4)).reshape(-1, 128)
    lg = F @ P["Wd"] + P["bd"]; lg -= lg.max(1, keepdims=True); p = np.exp(lg); return A, p / p.sum(1, keepdims=True)

# pick 24 grid digits: 2 per class + 4 that were hard early on
mid = fwd(snaps[keys[len(keys) // 3]], Cva)[1].argmax(1)
grid = []
for c in range(10): grid += list(np.where(yva == c)[0][:2])
hard = [i for i in np.random.default_rng(2).permutation(len(yva)) if i not in grid][:4]
grid = grid + hard
rng = np.random.default_rng(5); rng.shuffle(grid)
final = fwd(snaps[keys[-1]], Cva)[1].argmax(1)
EX = [i for i in np.where((yva == 8) & (final == 8))[0]][0]

def digit_img(x, size):
    a = (x * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(a).resize((size, size), Image.NEAREST).convert("RGB")
    arr = np.array(im).astype(np.float32) / 255
    out = np.array(PANEL, np.float32) * (1 - arr) + np.array(WHITE, np.float32) * arr
    return Image.fromarray(out.astype(np.uint8))

def signed_img(m, size, scale):
    v = np.clip(m / (scale + 1e-9), -1, 1)
    base = np.array((18, 26, 35), np.float32)
    pos = np.array(GOLD, np.float32); neg = np.array(BLUE, np.float32)
    rgb = base + (np.where(v[..., None] > 0, pos, neg) - base) * np.abs(v[..., None]) ** 0.8
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.NEAREST)

fH = R.font(R.JB, 24, 500); fLab = R.font(R.JB, 28, 600); fBar = R.font(R.JB, 24, 600)
def header(d):
    d.text((W / 2, 112), f"REDES NEURONALES · EP. 2 · {HANDLE}", font=R.fMonoS, fill=(70, 84, 98), anchor="mm")
    d.text((W / 2, 205), "¿puede una red neuronal", font=R.fTitle, fill=GOLD, anchor="mm")
    pill = "aprender a ver"
    tb = d.textbbox((0, 0), pill, font=R.fPill); tw = tb[2] - tb[0]
    x0 = W / 2 - (tw + 60) / 2 - 22; y0 = 262
    d.rounded_rectangle((x0, y0, x0 + tw + 60, y0 + 86), radius=12, fill=(22, 30, 40), outline=(62, 78, 96), width=2)
    d.text((x0 + 30 + tw / 2, y0 + 43), pill, font=R.fPill, fill=WHITE, anchor="mm")
    d.text((x0 + tw + 60 + 34, y0 + 42), "?", font=R.fQ, fill=GOLD, anchor="mm")

LX0, LX1, LY0, LY1 = 150, 990, 1560, 1765
lmin, lmax = np.log10(0.003), np.log10(4.0)
def lpx(ep, v): return LX0 + (ep - 1) / (E - 1) * (LX1 - LX0), LY1 - (np.log10(max(v, 1e-4)) - lmin) / (lmax - lmin) * (LY1 - LY0)
def loss_panel(d, ep):
    R.panel(d, (60, 1440, 1020, 1840))
    d.text((90, 1478), "cross-entropy (escala log)", font=R.fMono, fill=GREY, anchor="lm")
    tr, va = L[ep - 1, :2]
    d.line((720, 1470, 750, 1470), fill=BLUE, width=4); d.text((762, 1470), f"train {tr:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    d.line((720, 1502, 750, 1502), fill=R.VALC, width=4); d.text((762, 1502), f"valid {va:.4f}", font=R.fMonoS, fill=WHITE, anchor="lm")
    for v, lab in [(1, "1"), (0.1, "0.1"), (0.01, "0.01")]:
        _, y = lpx(1, v); d.line((LX0, y, LX1, y), fill=(26, 36, 46), width=1); d.text((LX0 - 14, y), lab, font=R.fMonoS, fill=GREY, anchor="rm")
    d.line((LX0, LY1, LX1, LY1), fill=EDGE, width=2)
    for e in (0, 100, 200, 300, 400):
        x = LX0 + e / E * (LX1 - LX0); d.text((min(x, LX1 - 20), LY1 + 26), str(e), font=R.fMonoS, fill=GREY, anchor="mm")
    d.text(((LX0 + LX1) / 2, LY1 + 56), "epoch", font=R.fMonoS, fill=GREY, anchor="mm")
    for col, j in ((R.VALC, 1), (BLUE, 0)):
        pts = [lpx(e, L[e - 1, j]) for e in range(1, ep + 1)]
        if len(pts) > 1: d.line(pts, fill=col, width=3, joint="curve")
        d.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)

EXIMG = digit_img(Xva[EX], 220)
GRIDIMG = {i: digit_img(Xva[i], 76) for i in grid}

def train_frame(fi):
    k = keys[min(len(keys) - 1, int(round(fi / (TRAIN_F - 1) * (len(keys) - 1))))]
    P = snaps[k]
    img = R.BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA"); header(d)
    R.panel(d, (60, 380, 1020, 1410))
    d.text((92, 425), f"epoch {k} / {E}", font=R.fMono, fill=GREY, anchor="lm")
    d.text((988, 425), "Conv3×3(8) · MaxPool · Dense(10)", font=R.fMono, fill=GREY, anchor="rm")
    d.line((80, 452, 1000, 452), fill=(26, 36, 46), width=1)
    # --- what the network sees ---
    A, p = fwd(P, Cva[[EX] + grid])
    d.text((95, 482), "entrada", font=fH, fill=GREY, anchor="lm")
    img.paste(EXIMG, (95, 500)); d.rectangle((95, 500, 315, 720), outline=EDGE, width=2)
    d.text((370, 482), "filtros 3×3  →  mapas de activación", font=fH, fill=GREY, anchor="lm")
    Wc = P["Wc"]; ws = np.abs(Wc).max(); amax = A[0].max()
    for f in range(8):
        r, c = divmod(f, 4); x = 370 + c * 158; y = 506 + r * 116
        img.paste(signed_img(Wc[:, f].reshape(3, 3), 42, ws), (x, y + 31))
        d.text((x + 46, y + 52), "→", font=fH, fill=GREY, anchor="lm")
        img.paste(signed_img(A[0][..., f], 100, amax), (x + 66, y))
        d.rectangle((x + 66, y, x + 166, y + 100), outline=EDGE, width=1)
    # probability bars
    probs = p[0]; best = probs.argmax()
    d.text((95, 762), "¿qué número es?", font=fH, fill=GREY, anchor="lm")
    for c in range(10):
        x = 120 + c * 86; h = probs[c] * 92; col = (GREEN if c == 8 else REDC) if c == best else (60, 74, 88)
        d.rounded_rectangle((x, 880 - max(h, 3), x + 52, 880), radius=5, fill=col)
        d.text((x + 26, 900), str(c), font=fBar, fill=WHITE if c == best else GREY, anchor="mm")
        if c == best: d.text((x + 26, 880 - h - 18), f"{probs[c] * 100:.0f}%", font=R.fMonoS, fill=WHITE, anchor="mm")
    d.line((80, 930, 1000, 930), fill=(26, 36, 46), width=1)
    # --- validation grid ---
    acc = L[k - 1, 3]
    d.text((95, 960), "dígitos que nunca ha visto", font=fH, fill=GREY, anchor="lm")
    d.text((988, 960), f"acc valid {acc * 100:.1f}%", font=R.fMono, fill=GOLD if acc > 0.95 else WHITE, anchor="rm")
    pred = p[1:].argmax(1)
    for j, i in enumerate(grid):
        r, c = divmod(j, 8); x = 96 + c * 112; y = 990 + r * 132
        ok = pred[j] == yva[i]; col = GREEN if ok else REDC
        img.paste(GRIDIMG[i], (x + 9, y)); d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((x + 8, y - 1, x + 86, y + 77), outline=col, width=3)
        d.text((x + 47, y + 98), f"→ {pred[j]}", font=fLab, fill=col, anchor="mm")
    loss_panel(d, k)
    return img

fSub = R.font(R.SG, 44, 500); fFollow = R.font(R.JB, 26, 600); fHandle = R.font(R.SG, 50, 700); fBrand = R.font(R.SG, 96, 700)
def end_card():
    img = R.BGIMG.copy(); d = ImageDraw.Draw(img, "RGBA")
    # icon: stacked feature maps shrinking -> dense node column
    glowl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); gd = ImageDraw.Draw(glowl)
    shapes = []
    for s, x0, n in ((150, 250, 3), (100, 470, 4), (60, 650, 5)):
        for j in range(n): shapes.append((x0 + j * 16, 690 - s / 2 - j * 12, x0 + j * 16 + s, 690 + s / 2 - j * 12))
    for b in shapes: gd.rectangle(b, outline=GOLD + (160,), width=6)
    nodes = [(840, 590 + j * 50) for j in range(5)]
    for x, y in nodes: gd.ellipse((x - 22, y - 22, x + 22, y + 22), fill=GOLD + (140,))
    img.alpha_composite(glowl.filter(ImageFilter.GaussianBlur(12))); d = ImageDraw.Draw(img, "RGBA")
    for b in shapes: d.rectangle(b, fill=(20, 26, 34), outline=GOLD, width=3)
    for x, y in nodes:
        d.line((735, 640, x - 16, y), fill=GOLD + (100,), width=2)
    for x, y in nodes: d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=GOLD, outline=(255, 226, 160), width=2)
    tl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(tl).text((W / 2, 940), "Redes Neuronales", font=fBrand, fill=GOLD, anchor="mm")
    img.alpha_composite(tl.filter(ImageFilter.GaussianBlur(10))); img.alpha_composite(tl); d = ImageDraw.Draw(img, "RGBA")
    d.text((W / 2, 1030), "Ep. 2 · Redes convolucionales", font=fSub, fill=WHITE, anchor="mm")
    R.panel(d, (200, 1120, 880, 1380))
    d.text((W / 2, 1180), "PRÓXIMO EPISODIO: RNN / LSTM", font=fFollow, fill=GOLD, anchor="mm")
    d.text((W / 2, 1250), "sígueme para no perdértelo", font=R.fMono, fill=GREY, anchor="mm")
    d.text((W / 2, 1318), HANDLE, font=fHandle, fill=WHITE, anchor="mm")
    return img

if __name__ == "__main__":
    import sys
    END = end_card(); LAST = train_frame(TRAIN_F - 1)
    def frame(f):
        if f < TRAIN_F - 1: return train_frame(f)
        if f < TRAIN_F + HOLD_F: return LAST
        return Image.blend(LAST, END, min(1, (f - TRAIN_F - HOLD_F) / 14))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1:]): frame(f).convert("RGB").save(f"work/c{f}.png")
        sys.exit()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "work/cnn.mp4"], stdin=subprocess.PIPE)
    for f in range(N): p.stdin.write(frame(f).convert("RGB").tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "work/cnn.mp4", "-i", "work/mlp.wav", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "160k", "-shortest", "out/ep2_cnn_aprender_a_ver.mp4"])
