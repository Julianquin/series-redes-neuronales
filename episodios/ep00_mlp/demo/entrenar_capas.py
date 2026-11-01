"""Entrena la red del episodio 00 sobre los datos del invernadero y guarda sus pesos.

Las escenas animan la deformación con ESTOS pesos, así que lo que se ve en el vídeo es la
transformación que la red aprendió de verdad, no una inventada.

Resultado (semillas 0-9, 8000 pasos): una capa de dos neuronas se queda en ~10 % de error;
dos capas llegan a 0 %. Es el argumento de s05: hay que encadenar capas.

    .venv/bin/python episodios/ep00_mlp/demo/entrenar_capas.py
"""
import json
from pathlib import Path

import numpy as np

from lib.componentes.nube_puntos import NubePuntos

SALIDA = Path(__file__).parent / "pesos_invernadero.json"


def datos(n: int = 200, escala: float = 1.6, semilla: int = 0):
    """Los mismos puntos que verá el espectador, en coordenadas de escena."""
    nube = NubePuntos.lunas(n=n, escala=escala, semilla=semilla)
    return nube.xy_escena, nube.clases.astype(float)


def entrenar(X, y, capas: int, semilla: int, pasos: int = 8000, lr: float = 0.2, momento: float = 0.9):
    """MLP 2→(2 ReLU)×capas→1 con descenso de gradiente y momento, en numpy."""
    rng = np.random.default_rng(semilla)
    P = {"v": rng.normal(0, 1, 2), "c": np.zeros(1)}
    for i in range(1, capas + 1):
        P[f"W{i}"], P[f"b{i}"] = rng.normal(0, 1, (2, 2)), rng.normal(0, 0.3, 2)
    m = {k: np.zeros_like(v) for k, v in P.items()}

    def adelante(X):
        activaciones, Z = [X], None
        for i in range(1, capas + 1):
            Z = activaciones[-1] @ P[f"W{i}"].T + P[f"b{i}"]
            activaciones.append(np.maximum(Z, 0))
        return activaciones

    for _ in range(pasos):
        A = adelante(X)
        p = 1 / (1 + np.exp(-(A[-1] @ P["v"] + P["c"])))
        g = (p - y) / len(X)
        gr = {"v": A[-1].T @ g, "c": np.array([g.sum()])}
        gA = np.outer(g, P["v"])
        for i in range(capas, 0, -1):
            gA = gA * (A[i] > 0)                      # derivada de la ReLU
            gr[f"W{i}"], gr[f"b{i}"] = gA.T @ A[i - 1], gA.sum(0)
            gA = gA @ P[f"W{i}"]
        for k in P:
            m[k] = momento * m[k] + gr[k]
            P[k] -= lr * m[k]

    A = adelante(X)
    error = float(((A[-1] @ P["v"] + P["c"] > 0) != y).mean())
    return error, P


def main():
    X, y = datos()
    resultado = {}
    for capas in (1, 2):
        error, P = min((entrenar(X, y, capas, s) for s in range(10)), key=lambda r: r[0])
        resultado[f"{capas}_capa{'s' if capas > 1 else ''}"] = {
            "error": round(error, 4), **{k: np.round(v, 4).tolist() for k, v in P.items()}}
        print(f"{capas} capa(s) de 2 neuronas → error {error:.3f}")
    SALIDA.write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(f"pesos guardados en {SALIDA.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
