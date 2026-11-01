import numpy as np, json, pickle, time, sys
rng = np.random.default_rng(11)
raw = json.load(open('data/municipios.json'))
recs = [x for x in raw if not any(ch in x['n'] for ch in ',(/') and 3 <= len(x['n']) <= 22]
COMM = sorted(set(x['c'] for x in recs)); cidx = {c: i for i, c in enumerate(COMM)}
chars = sorted(set(''.join(x['n'] for x in recs))); START, END = '^', '$'
vocab = chars + [START, END]; ix = {c: i for i, c in enumerate(vocab)}; V = len(vocab); C = len(COMM); Hd = 128
enc = [([ix[START]] + [ix[c] for c in x['n']] + [ix[END]], cidx[x['c']]) for x in recs]
print(len(recs), 'nombres', V, 'símbolos', C, 'comunidades')
P = dict(Wt=rng.normal(0, 0.1, (V, 4 * Hd)), Wc=rng.normal(0, 0.1, (C, 4 * Hd)), Wh=rng.normal(0, 1 / np.sqrt(Hd), (Hd, 4 * Hd)),
         b=np.zeros(4 * Hd), Wy=rng.normal(0, 1 / np.sqrt(Hd), (Hd, V)), by=np.zeros(V))
P['b'][Hd:2 * Hd] = 1.0
sig = lambda x: 1 / (1 + np.exp(-x))

def cell(P, tok, cm, h, c):
    z = P['Wt'][tok] + P['Wc'][cm] + h @ P['Wh'] + P['b']
    i, f, o, g = sig(z[:, :Hd]), sig(z[:, Hd:2 * Hd]), sig(z[:, 2 * Hd:3 * Hd]), np.tanh(z[:, 3 * Hd:])
    c2 = f * c + i * g; h2 = o * np.tanh(c2)
    lg = h2 @ P['Wy'] + P['by']; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    return h2, c2, p, (i, f, o, g, c, c2)

def generate(P, comm, u, temp=0.8, maxlen=24):
    """u: array (maxlen,) of uniforms -> same randomness across checkpoints."""
    h = np.zeros((1, Hd)); c = np.zeros((1, Hd)); tok = np.array([ix[START]]); cm = np.array([comm]); out = ''
    for s in range(maxlen):
        h, c, p, _ = cell(P, tok, cm, h, c); q = p[0] ** (1 / temp); q /= q.sum()
        k = int(np.searchsorted(np.cumsum(q), u[s])); k = min(k, V - 1)
        if vocab[k] == END: break
        out += vocab[k]; tok = np.array([k])
    return out

if __name__ == '__main__':
    ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; B, lr = 64, 0.004
    m = {k: 0 for k in P}; v = {k: 0 for k in P}; L = []
    CK = [1, 60, 250, 800, ITERS]; snaps = {}; t0 = time.time()
    for it in range(1, ITERS + 1):
        batch = [enc[j] for j in rng.integers(0, len(enc), B)]
        T = max(len(s) for s, _ in batch) - 1
        tk = np.full((B, T + 1), ix[END]); mask = np.zeros((B, T))
        for r, (s, _) in enumerate(batch): tk[r, :len(s)] = s; mask[r, :len(s) - 1] = 1
        cm = np.array([c for _, c in batch])
        h = np.zeros((B, Hd)); c = np.zeros((B, Hd)); cache = []; loss = 0; n = mask.sum()
        for t in range(T):
            hp = h; h, c, p, cc = cell(P, tk[:, t], cm, h, c); cache.append((hp, h, p, cc))
            loss -= (np.log(p[np.arange(B), tk[:, t + 1]] + 1e-9) * mask[:, t]).sum()
        L.append(loss / n)
        g = {k: np.zeros_like(P[k]) for k in P}; dh_n = np.zeros((B, Hd)); dc_n = np.zeros((B, Hd))
        for t in reversed(range(T)):
            hp, h, p, (i, f, o, gg, cp, c2) = cache[t]
            dy = p.copy(); dy[np.arange(B), tk[:, t + 1]] -= 1; dy *= mask[:, t][:, None] / n
            g['Wy'] += h.T @ dy; g['by'] += dy.sum(0)
            dh = dy @ P['Wy'].T + dh_n; tc = np.tanh(c2); dc = dh * o * (1 - tc ** 2) + dc_n
            dz = np.concatenate([dc * gg * i * (1 - i), dc * cp * f * (1 - f), dh * tc * o * (1 - o), dc * i * (1 - gg ** 2)], 1)
            g['Wh'] += hp.T @ dz; g['b'] += dz.sum(0); np.add.at(g['Wt'], tk[:, t], dz); np.add.at(g['Wc'], cm, dz)
            dh_n = dz @ P['Wh'].T; dc_n = dc * f
        lr_t = lr * (0.15 + 0.85 * 0.5 * (1 + np.cos(np.pi * it / ITERS)))
        for k in P:
            gk = np.clip(g[k], -1, 1)
            m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk ** 2
            P[k] = P[k] - lr_t * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
        if it in CK: snaps[it] = {k: P[k].copy() for k in P}
        if it % 100 == 0: print(it, round(float(np.mean(L[-100:])), 3), f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(dict(L=np.array(L), snaps=snaps, CK=CK, vocab=vocab, COMM=COMM, recs=recs, Hd=Hd), open('work/run_pueblos.pkl', 'wb'))
    print('hecho')
