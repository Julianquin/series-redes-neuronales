"""Reproduce data/pueblos_sel.json (Ep. 3b, pueblos que no existen).
Los nombres de las rondas se ELIGIERON A MANO entre muchas muestras: aquí quedan fijados y se comprueba que no existen.
Requiere work/run_pueblos.pkl (python3 train_pueblos.py 3500)."""
import pickle, json, difflib, numpy as np
import train_pueblos as T
r = pickle.load(open('work/run_pueblos.pkl', 'rb'))
COMM, recs, CK = r['COMM'], r['recs'], r['CK']; real = set(x['n'].lower() for x in recs); P = r['snaps'][CK[-1]]
def nearest(name, com):
    pool = [x for x in recs if x['c'] == com]; m = difflib.get_close_matches(name, [x['n'] for x in pool], n=1, cutoff=0.0)[0]
    return m, next(x['p'] for x in pool if x['n'] == m)
def prov(n, com): return next(x['p'] for x in recs if x['n'] == n and x['c'] == com)
# Parte 1: evolución en Andalucía con la MISMA secuencia aleatoria en cada checkpoint (semilla 5, sorteo nº 6)
ci = COMM.index('Andalucía'); rng = np.random.default_rng(5)
for _ in range(7): u = rng.random(24)
evo = [T.generate(r['snaps'][k], ci, u, 0.8) for k in CK]; nm, npv = nearest(evo[-1], 'Andalucía')
# Rondas del reto (elección manual): 2 reales + 1 inventado; 'pos' = posición del inventado (0=A, 1=B, 2=C)
ROUNDS = [('Cataluña', ['Castellolí', 'Santa Pau'], 'Vilanuïga', 1), ('País Vasco', ['Ispaster', 'Albiztur'], 'Mukieta', 2)]
rounds = []
for com, reals, inv, pos in ROUNDS:
    assert inv.lower() not in real, f'{inv} existe de verdad'
    items = [{'name': n, 'real': True, 'prov': prov(n, com)} for n in reals]
    near = nearest(inv, com); items.insert(pos, {'name': inv, 'real': False, 'near': near[0], 'prov': near[1]}); rounds.append({'comm': com, 'items': items})
# % de nombres generados que NO existen (Andalucía, temperatura 0,8)
rg = np.random.default_rng(0); gen = [T.generate(P, ci, rg.random(24), 0.8) for _ in range(500)]
novel = 100 * float(np.mean([g.lower() not in real for g in gen]))
sel = dict(comm1='Andalucía', evo=evo, near_name=nm, near_prov=npv, rounds=rounds, nparams=int(sum(v.size for v in P.values())),
           ncomm=len(COMM), nnames=len(recs), novel_pct=round(novel, 1))
json.dump(sel, open('data/pueblos_sel.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(sel, ensure_ascii=False)[:300])
