import numpy as np, pickle, networkx as nx
G = nx.karate_club_graph(); n = G.number_of_nodes()
y = np.array([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(n)])
A = nx.to_numpy_array(G) > 0; A = A.astype(float) + np.eye(n); dg = A.sum(1); Ah = A / np.sqrt(np.outer(dg, dg))
X = np.eye(n); LAB = [0, 33]
rng = np.random.default_rng(4)
P = dict(W1=rng.normal(0, 0.3, (n, 16)), W2=rng.normal(0, 0.4, (16, 2)), W3=rng.normal(0, 0.5, (2, 2)), b3=np.zeros(2))
def fwd(P):
    Z1 = Ah @ X @ P['W1']; H1 = np.tanh(Z1); Z2 = Ah @ H1 @ P['W2']; H2 = np.tanh(Z2)
    lg = H2 @ P['W3'] + P['b3']; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
    return H1, H2, p
ITERS, lr = 200, 0.01; m = {k: 0 for k in P}; v = {k: 0 for k in P}; L = []; snaps = {}
for it in range(1, ITERS + 1):
    H1, H2, p = fwd(P)
    snaps[it] = {k: P[k].copy() for k in P}
    dl = np.zeros_like(p); dl[LAB] = p[LAB]; dl[LAB, y[LAB]] -= 1; dl /= len(LAB)
    g = {'W3': H2.T @ dl, 'b3': dl.sum(0)}
    dZ2 = (dl @ P['W3'].T) * (1 - H2 ** 2); g['W2'] = (Ah @ H1).T @ dZ2
    dZ1 = (Ah.T @ dZ2 @ P['W2'].T) * (1 - H1 ** 2); g['W1'] = (Ah @ X).T @ dZ1 + 5e-4 * P['W1']
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
        P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v[k] / (1 - 0.999 ** it)) + 1e-8)
    L.append((-np.log(p[LAB, y[LAB]]).mean(), (p.argmax(1) == y).mean()))
    if it % 25 == 0: print(it, L[-1])
pos = nx.kamada_kawai_layout(G); pos = np.array([pos[i] for i in range(n)])
pickle.dump(dict(L=np.array(L), snaps=snaps, y=y, Ah=Ah, edges=list(G.edges()), pos=pos, LAB=LAB), open('work/run_gnn.pkl', 'wb'))
