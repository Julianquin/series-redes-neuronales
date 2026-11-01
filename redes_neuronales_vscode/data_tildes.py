"""Datos: palabras del español sin tilde -> posición de la tilde (forma más frecuente según wordfreq)."""
import re, json, numpy as np
from wordfreq import top_n_list, zipf_frequency
STRIP = str.maketrans("áéíóú", "aeiou")
ALPHA = "abcdefghijklmnopqrstuvwxyzñ"; IX = {c: i + 1 for i, c in enumerate(ALPHA)}; PAD = 0; TMAX = 16
def build(n=60000, seed=0):
    words = [w for w in top_n_list("es", n) if re.fullmatch(r"[a-zñáéíóú]{2,16}", w) and sum(c in "áéíóú" for c in w) <= 1]
    forms = {}
    for w in words: forms.setdefault(w.translate(STRIP), []).append((zipf_frequency(w, "es"), w))
    keys, targets, amb = [], [], []
    for k, fs in forms.items():
        fs.sort(reverse=True); keys.append(k); targets.append(fs[0][1])
        amb.append([w for f, w in fs if f >= fs[0][0] - 1.0])        # formas con ≥10 % de la frecuencia de la ganadora
    rng = np.random.default_rng(seed); idx = rng.permutation(len(keys)); nte = len(keys) // 10
    return dict(keys=keys, targets=targets, amb=amb, train=idx[nte:].tolist(), test=idx[:nte].tolist())
def encode(keys, targets):
    n = len(keys); X = np.zeros((n, TMAX), np.int64); Y = np.zeros((n, TMAX), np.int64); M = np.zeros((n, TMAX), bool)
    for r, (k, t) in enumerate(zip(keys, targets)):
        for i, (c, ct) in enumerate(zip(k, t)): X[r, i] = IX[c]; Y[r, i] = int(ct in "áéíóú"); M[r, i] = True
    return X, Y, M
if __name__ == "__main__":
    D = build(); json.dump(D, open("data/tildes_data.json", "w"), ensure_ascii=False)
    t = np.array([any(c in "áéíóú" for c in w) for w in D["targets"]]); a = np.array([len(x) > 1 for x in D["amb"]])
    print(len(D["keys"]), "palabras ·", round(t.mean() * 100, 1), "% con tilde ·", round(a.mean() * 100, 1), "% ambiguas sin contexto")
    for k in ["murcielago", "camion", "arbol", "termino", "esta", "cancion", "telefono", "rapido"]:
        if k in D["keys"]: i = D["keys"].index(k); print(k, "->", D["targets"][i], D["amb"][i])
