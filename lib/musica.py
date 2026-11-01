"""Análisis de pistas de música: tempo, energía y transiciones.

El resultado se guarda en assets/musica/analisis/<pista>.json y lo usa e6 para que una
transición de la música (p. ej. la entrada de la pista completa tras la intro) caiga justo en
un cambio de bloque del vídeo.

    .venv/bin/python -m lib.musica assets/musica/*.mp3      # analiza y muestra la estructura
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from lib import config

DIR_ANALISIS = config.RAIZ / "assets" / "musica" / "analisis"
VENTANA_S = 4.0          # energía media por ventanas de 4 s
SALTO_MIN_DB = 6.0       # una "subida" es un aumento de energía de al menos 6 dB entre ventanas


def _huella(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def _sonoridad(ruta: Path) -> float:
    """Sonoridad integrada de la pista (LUFS): e6 la iguala entre pistas bajo la voz."""
    import re
    import subprocess

    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(ruta), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    resumen = r.stderr[r.stderr.rfind("Summary:"):]
    return float(re.search(r"I:\s*(-?[\d.]+) LUFS", resumen).group(1))


def analizar(ruta: Path) -> dict:
    """Tempo, pulsos, energía por ventanas, subidas y bajadas de energía y límites de sección
    (por timbre). Todos los instantes se ajustan al pulso más cercano. Se cachea por huella."""
    DIR_ANALISIS.mkdir(parents=True, exist_ok=True)
    cache = DIR_ANALISIS / f"{ruta.stem}.json"
    huella = _huella(ruta)
    if cache.exists():
        datos = json.loads(cache.read_text(encoding="utf-8"))
        if datos.get("huella") == huella and "lufs" in datos:
            return datos

    import librosa

    y, sr = librosa.load(str(ruta), sr=22050, mono=True)
    duracion = len(y) / sr
    tempo, pulsos = librosa.beat.beat_track(y=y, sr=sr)
    t_pulsos = librosa.frames_to_time(pulsos, sr=sr)

    def al_pulso(t: float) -> float:
        return round(float(t_pulsos[np.argmin(np.abs(t_pulsos - t))]), 2) if len(t_pulsos) else round(t, 2)

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t_rms = librosa.times_like(rms, sr=sr, hop_length=hop)
    ventanas = [(float(a), float(20 * np.log10(rms[(t_rms >= a) & (t_rms < a + VENTANA_S)].mean() + 1e-9)))
                for a in np.arange(0, duracion - VENTANA_S / 2, VENTANA_S)]

    # Límites de sección por timbre; las subidas/bajadas de energía se sitúan en el límite más cercano.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    croma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rasgos = librosa.util.normalize(np.vstack([mfcc, croma]), axis=1)
    limites = librosa.frames_to_time(librosa.segment.agglomerative(rasgos, max(4, int(duracion // 20))),
                                     sr=sr, hop_length=hop)
    limites = sorted({al_pulso(t) for t in limites if 1.0 < t < duracion - 1.0})

    def situar(t_aprox: float) -> float:
        cercanos = [l for l in limites if abs(l - t_aprox) <= VENTANA_S]
        return min(cercanos, key=lambda l: abs(l - t_aprox)) if cercanos else al_pulso(t_aprox)

    subidas, bajadas = [], []
    for (a0, d0), (a1, d1) in zip(ventanas, ventanas[1:]):
        if d1 - d0 >= SALTO_MIN_DB:
            subidas.append({"t": situar(a1), "db": round(d1 - d0, 1)})
        elif d0 - d1 >= SALTO_MIN_DB:
            bajadas.append({"t": situar(a1), "db": round(d0 - d1, 1)})

    def sin_duplicados(eventos: list[dict]) -> list[dict]:
        """Dos ventanas pueden caer en el mismo límite de sección: se queda el salto mayor."""
        mejores: dict[float, dict] = {}
        for e in eventos:
            if e["t"] not in mejores or e["db"] > mejores[e["t"]]["db"]:
                mejores[e["t"]] = e
        return sorted(mejores.values(), key=lambda e: e["t"])

    subidas, bajadas = sin_duplicados(subidas), sin_duplicados(bajadas)
    lufs = _sonoridad(ruta)

    datos = {
        "pista": str(ruta.relative_to(config.RAIZ)), "huella": huella, "lufs": lufs, "duracion": round(duracion, 2),
        "tempo": round(float(np.atleast_1d(tempo)[0]), 1),
        "subidas": subidas, "bajadas": bajadas, "limites": limites,
        "energia_db": [[round(a, 1), round(d, 1)] for a, d in ventanas],
    }
    cache.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")
    return datos


def primera_subida(datos: dict) -> float | None:
    """La transición más útil: la primera gran subida de energía (fin de la intro)."""
    return datos["subidas"][0]["t"] if datos["subidas"] else None


def main(rutas: list[str]) -> None:
    for r in rutas:
        ruta = Path(r).resolve()
        d = analizar(ruta)
        print(f"== {ruta.name} · {d['duracion']:.1f} s · {d['tempo']:.0f} BPM · {d['lufs']:.1f} LUFS")
        print("   subidas de energía:", ", ".join(f"{s['t']:.1f} s (+{s['db']} dB)" for s in d["subidas"]) or "ninguna")
        print("   bajadas de energía:", ", ".join(f"{s['t']:.1f} s (−{s['db']} dB)" for s in d["bajadas"]) or "ninguna")
        print("   límites de sección:", ", ".join(f"{t:.1f}" for t in d["limites"]))


if __name__ == "__main__":
    main(sys.argv[1:])
