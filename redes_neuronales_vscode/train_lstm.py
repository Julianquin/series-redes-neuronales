import numpy as np, pickle
TEXT = ("la memoria es lo que queda cuando todo pasa. una red recurrente lee una letra, "
        "guarda algo de lo que ha visto y lo usa para adivinar la siguiente. letra a letra, "
        "paso a paso, aprende que después de la q viene la u, que las palabras tienen final "
        "y que una frase empieza donde otra termina. ")
chars = sorted(set(TEXT)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in TEXT]); n = len(data)
rng = np.random.default_rng(0)
Hd = 96; D = V + Hd
P = dict(W=rng.normal(0, 1 / np.sqrt(D), (D, 4 * Hd)), b=np.zeros(4 * Hd), Wy=rng.normal(0, 1 / np.sqrt(Hd), (Hd, V)), by=np.zeros(V))
P['b'][Hd:2 * Hd] = 1.0
sig = lambda x: 1 / (1 + np.exp(-x))
def step(P, x, h, c):
    xh = np.concatenate([x, h], 1); z = xh @ P['W'] + P['b']
    i, f, o, g = sig(z[:, :Hd]), sig(z[:, Hd:2*Hd]), sig(z[:, 2*Hd:3*Hd]), np.tanh(z[:, 3*Hd:])
    c2 = f * c + i * g; h2 = o * np.tanh(c2)
    lg = h2 @ P['Wy'] + P['by']; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    return h2, c2, p, (xh, i, f, o, g, c, c2)
T, B, ITERS, lr = 40, 32, 500, 0.004
m = {k: 0 for k in P}; v = {k: 0 for k in P}; L = []
CK = [1, 40, 90, 160, 500]; snaps = {}
for it in range(1, ITERS + 1):
    st = rng.integers(0, n, B); idx = data[(st[:, None] + np.arange(T + 1)) % n]
    X = np.eye(V)[idx[:, :-1]]; Y = idx[:, 1:]
    h = np.zeros((B, Hd)); c = np.zeros((B, Hd)); cache = []; loss = 0
    for t in range(T):
        h, c, p, cc = step(P, X[:, t], h, c); cache.append((h, p, cc)); loss -= np.log(p[np.arange(B), Y[:, t]] + 1e-9).mean()
    L.append(loss / T)
    g = {k: np.zeros_like(P[k]) for k in P}; dh_n = np.zeros((B, Hd)); dc_n = np.zeros((B, Hd))
    for t in reversed(range(T)):
        h, p, (xh, i, f, o, gg, cp, c2) = cache[t]
        dy = p.copy(); dy[np.arange(B), Y[:, t]] -= 1; dy /= (B * T)
        g['Wy'] += h.T @ dy; g['by'] += dy.sum(0)
        dh = dy @ P['Wy'].T + dh_n; tc = np.tanh(c2)
        dc = dh * o * (1 - tc ** 2) + dc_n
        dz = np.concatenate([dc * gg * i * (1 - i), dc * cp * f * (1 - f), dh * tc * o * (1 - o), dc * i * (1 - gg ** 2)], 1)
        g['W'] += xh.T @ dz; g['b'] += dz.sum(0)
        dxh = dz @ P['W'].T; dh_n = dxh[:, V:]; dc_n = dc * f
    for k in P:
        gk = np.clip(g[k], -1, 1)
        m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk ** 2
        P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
    if it in CK: snaps[it] = {k: P[k].copy() for k in P}
    if it % 500 == 0: print(it, L[-1])
# generation per checkpoint
SEED = "la memoria "; GEN = 150; gens = {}
for k, Pk in snaps.items():
    r = np.random.default_rng(42); h = np.zeros((1, Hd)); c = np.zeros((1, Hd))
    for ch in SEED[:-1]: h, c, p, _ = step(Pk, np.eye(V)[[ix[ch]]], h, c)
    cur = ix[SEED[-1]]; out = []
    for s in range(GEN):
        h, c, p, _ = step(Pk, np.eye(V)[[cur]], h, c); p = p[0]
        q = p ** (1 / 0.5); q /= q.sum(); nxt = r.choice(V, p=q)
        top = np.argsort(p)[::-1][:5]
        out.append(dict(ch=chars[nxt], prob=p[nxt], top=[(chars[j], p[j]) for j in top], h=h[0].copy(), c=c[0].copy()))
        cur = nxt
    gens[k] = out
    print(k, repr("".join(o['ch'] for o in out)))
pickle.dump(dict(L=np.array(L), gens=gens, CK=CK, SEED=SEED, V=V, Hd=Hd), open('work/run_lstm.pkl', 'wb'))
