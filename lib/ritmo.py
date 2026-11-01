"""Modelo del ritmo del narrador: cuánto dura una frase según su longitud.

Una frase larga se dice más rápido por carácter que una corta (los márgenes pesan menos), así que
un umbral fijo de segundos por carácter marca como sospechosas frases que están bien. El QA de e2
compara con este modelo lineal, ajustado sobre las frases ya aprobadas.

    .venv/bin/python -m lib.ritmo          # reajusta el modelo y muestra la línea para pipeline.yaml
"""
import glob
import json
import sys

import numpy as np

from lib import config
from lib.texto import numeros_a_palabras


def frases_aprobadas(motor: str | None = None) -> list[tuple[int, float]]:
    """(caracteres, duración) de las frases con QA aprobado, del motor indicado.

    Se filtra por motor porque cada uno habla a su ritmo: mezclarlos daría un modelo que no
    describe a ninguno.
    """
    datos = []
    for ruta in sorted(glob.glob(str(config.EPISODIOS / "*" / "build" / "audio" / "frases" / "*.json"))):
        d = json.loads(open(ruta, encoding="utf-8").read())
        if d["estado"] not in ("ok", "aceptada"):
            continue
        if motor and d.get("motor", "") != motor:
            continue
        datos.append((len(numeros_a_palabras(d["texto"])), d["duracion"]))
    return datos


def ajustar(datos: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Ajusta duración ≈ a + b·caracteres. Devuelve (a, b, r²)."""
    n = np.array([c for c, _ in datos], dtype=float)
    d = np.array([t for _, t in datos])
    b, a = np.polyfit(n, d, 1)
    return float(a), float(b), float(np.corrcoef(n, d)[0, 1] ** 2)


def main() -> int:
    motor = config.cargar("voz")["motor"]
    datos = frases_aprobadas(motor)
    print(f"motor: {motor}")
    if len(datos) < 10:
        print(f"[FALLO] solo {len(datos)} frases aprobadas de este motor: hacen falta al menos 10.\n"
              "   Genera un episodio con él antes de medir su ritmo.")
        return 1
    a, b, r2 = ajustar(datos)
    n = np.array([c for c, _ in datos], dtype=float)
    ratios = np.array([t for _, t in datos]) / (a + b * n)
    print(f"{len(datos)} frases · duración ≈ {a:.2f} s + {b:.4f}·caracteres · r² = {r2:.2f}")
    print(f"ratio observado: min {ratios.min():.2f} · p5 {np.percentile(ratios, 5):.2f} · "
          f"p95 {np.percentile(ratios, 95):.2f} · max {ratios.max():.2f}")
    print(f"\nEn config/pipeline.yaml → voz_qa:\n  duracion_modelo: [{a:.2f}, {b:.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
