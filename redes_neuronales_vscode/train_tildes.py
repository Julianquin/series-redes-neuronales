"""Transformer mínimo en NumPy que pone tildes. Capa: atención (1 cabeza) + residual + FFN + residual. Reanudable.
python3 train_tildes.py SEGUNDOS"""
import numpy as np, json, pickle, os, sys, time
from data_tildes import encode, TMAX, ALPHA
D = json.load(open("data/tildes_data.json"))
X, Y, M = encode(D["keys"], D["targets"]); TR, TE = np.array(D["train"]), np.array(D["test"])
LEN = M.sum(1); POSE = np.where(M, (LEN[:, None] - 1 - np.arange(TMAX)[None]), 0)       # distancia al final de la palabra
V = len(ALPHA) + 1; d, F, NL = 64, 128, 2
STATE = "work/tildes_state.pkl"; OUT = "work/run_tildes.pkl"
def init(rng):
    P = dict(E=rng.normal(0, .3, (V, d)), Ps=rng.normal(0, .3, (TMAX, d)), Pe=rng.normal(0, .3, (TMAX, d)), Wc=rng.normal(0, d ** -.5, (d, 2)), bc=np.zeros(2))
    for l in range(NL):
        for k, s in (("Wq", (d, d)), ("Wk", (d, d)), ("Wv", (d, d)), ("Wo", (d, d)), ("W1", (d, F)), ("W2", (F, d))): P[f"{k}{l}"] = rng.normal(0, s[0] ** -.5 * (0.5 if k in ("Wo", "W2") else 1), s)
        P[f"b1{l}"] = np.zeros(F); P[f"b2{l}"] = np.zeros(d)
    return P
def forward(P, xb, pe, mb, keep=False):
    h = P["E"][xb] + P["Ps"][None] + P["Pe"][pe]; cache = []; As = []
    neg = np.where(mb[:, None, :], 0.0, -1e9); s = d ** -.5
    for l in range(NL):
        Q, K, Vv = h @ P[f"Wq{l}"], h @ P[f"Wk{l}"], h @ P[f"Wv{l}"]
        S = Q @ K.transpose(0, 2, 1) * s + neg; S -= S.max(-1, keepdims=True); A = np.exp(S); A /= A.sum(-1, keepdims=True)
        Mt = A @ Vv; h1 = h + Mt @ P[f"Wo{l}"]; u = h1 @ P[f"W1{l}"] + P[f"b1{l}"]; z = np.maximum(u, 0); h2 = h1 + z @ P[f"W2{l}"] + P[f"b2{l}"]
        cache.append((h, Q, K, Vv, A, Mt, h1, u, z)); As.append(A); h = h2
    lg = h @ P["Wc"] + P["bc"]; lg -= lg.max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    return (p, cache, h, As) if keep else (p, As)
def mm(a, b): return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])
def grads(P, xb, yb, pe, mb):
    p, cache, hL, _ = forward(P, xb, pe, mb, True); n = mb.sum(); g = {}
    dl = p.copy(); bi, ti = np.nonzero(mb); dl[bi, ti, yb[bi, ti]] -= 1; dl *= mb[..., None] / n
    loss = -np.log(p[bi, ti, yb[bi, ti]] + 1e-12).mean()
    g["Wc"] = mm(hL, dl); g["bc"] = dl.sum((0, 1)); dh = dl @ P["Wc"].T; s = d ** -.5
    for l in reversed(range(NL)):
        h, Q, K, Vv, A, Mt, h1, u, z = cache[l]
        g[f"W2{l}"] = mm(z, dh); g[f"b2{l}"] = dh.sum((0, 1)); du = (dh @ P[f"W2{l}"].T) * (u > 0)
        g[f"W1{l}"] = mm(h1, du); g[f"b1{l}"] = du.sum((0, 1)); dh1 = dh + du @ P[f"W1{l}"].T
        g[f"Wo{l}"] = mm(Mt, dh1); dM = dh1 @ P[f"Wo{l}"].T
        dA = dM @ Vv.transpose(0, 2, 1); dV = A.transpose(0, 2, 1) @ dM; dS = A * (dA - (dA * A).sum(-1, keepdims=True))
        dQ = dS @ K * s; dK = dS.transpose(0, 2, 1) @ Q * s
        g[f"Wq{l}"] = mm(h, dQ); g[f"Wk{l}"] = mm(h, dK); g[f"Wv{l}"] = mm(h, dV)
        dh = dh1 + dQ @ P[f"Wq{l}"].T + dK @ P[f"Wk{l}"].T + dV @ P[f"Wv{l}"].T
    gE = np.zeros_like(P["E"]); np.add.at(gE, xb, dh); g["E"] = gE; g["Ps"] = dh.sum(0)
    gPe = np.zeros_like(P["Pe"]); np.add.at(gPe, pe, dh * mb[..., None]); g["Pe"] = gPe
    return g, loss
def evaluate(P, idx):
    out = []
    for i in range(0, len(idx), 2000):
        j = idx[i:i + 2000]; p, _ = forward(P, X[j], POSE[j], M[j]); out.append(p.argmax(-1))
    pred = np.concatenate(out); ok_pos = ((pred == Y[idx]) | ~M[idx]); return float(ok_pos.all(1).mean()), pred
if __name__ == "__main__":
    ITERS, B = 3500, 256; budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e12; t0 = time.time()
    keys = sorted(set(np.unique(np.round(ITERS * (np.arange(1, 121) / 120) ** 1.8).astype(int).clip(1, ITERS))) | {1})
    if os.path.exists(STATE): S = pickle.load(open(STATE, "rb"))
    else:
        rng = np.random.default_rng(0); P = init(rng); S = dict(rng=rng, P=P, m={k: 0 for k in P}, v={k: 0 for k in P}, it=0, L=[], snaps={}, acc={})
    rng, P = S["rng"], S["P"]; TEs = TE[:3000]
    while S["it"] < ITERS and time.time() - t0 < budget:
        S["it"] += 1; it = S["it"]; j = TR[rng.integers(0, len(TR), B)]
        g, loss = grads(P, X[j], Y[j], POSE[j], M[j]); lr = 2e-3 * min(1, it / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / ITERS)))
        for k in P:
            S["m"][k] = 0.9 * S["m"][k] + 0.1 * g[k]; S["v"][k] = 0.999 * S["v"][k] + 0.001 * g[k] ** 2
            P[k] = P[k] - lr * (S["m"][k] / (1 - 0.9 ** it)) / (np.sqrt(S["v"][k] / (1 - 0.999 ** it)) + 1e-8)
        if it in keys or it % 250 == 0:
            pv, _ = forward(P, X[TEs], POSE[TEs], M[TEs]); bi, ti = np.nonzero(M[TEs]); vl = -np.log(pv[bi, ti, Y[TEs][bi, ti]] + 1e-12).mean()
            S["L"].append((it, loss, vl)); acc, _ = evaluate(P, TEs); S["acc"][it] = acc
            if it in keys: S["snaps"][it] = {k: v.copy() for k, v in P.items()}
            if it % 500 == 0: print(it, round(loss, 4), round(vl, 4), "acierto", round(acc, 4), f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(S, open(STATE, "wb"))
    if S["it"] >= ITERS:
        acc, pred = evaluate(P, TE); pickle.dump(dict(snaps=S["snaps"], L=np.array(S["L"]), acc=S["acc"], final_acc=acc, ITERS=ITERS, d=d, F=F, NL=NL), open(OUT, "wb")); print("hecho · acierto examen completo:", round(acc, 4))
    else: print("pausa en", S["it"])
