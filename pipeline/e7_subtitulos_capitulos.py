"""Etapa 7 — Subtítulos .srt desde el guion y capítulos desde los bloques (§10.9).

Subtítulos: texto ORIGINAL del guion (nunca la transcripción de Whisper) con los tiempos exactos de
cada frase de e2, desplazados por la duración acumulada de los segmentos (la misma línea temporal
que usa e6). Líneas de máx. subtitulos.max_caracteres_linea y máx. subtitulos.max_lineas por
subtítulo; una frase que no cabe se reparte en varios, con tiempo proporcional a su longitud.

Capítulos: uno por bloque del guion (gancho, intuicion…), titulado con `titulo_capitulo` del primer
segmento del bloque. Reglas de YouTube: el primero en 0:00, al menos 3, y cada uno de ≥10 s
(los más cortos se funden con el anterior).

Salida: salida/epXX/<slug>.srt y salida/epXX/capitulos.txt

    python -m pipeline.e7_subtitulos_capitulos --ep 99
"""
import json
import sys
import textwrap

import soundfile as sf

from lib import config
from pipeline import comun

TITULOS_POR_DEFECTO = {
    "gancho": "Introducción", "limitacion": "El problema", "intuicion": "La idea",
    "matematicas": "Las matemáticas", "demo": "Demo", "cuando_usar": "Cuándo usarla", "puente": "Lo que viene",
}
CAPITULO_MIN_S = 10


def tiempo_srt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def tiempo_capitulo(t: float) -> str:
    s = int(t)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def partir_frase(texto: str, inicio: float, fin: float, max_car: int, max_lineas: int) -> list[tuple[float, float, str]]:
    """Reparte una frase en subtítulos de ≤max_lineas líneas de ≤max_car caracteres."""
    lineas = textwrap.wrap(texto, max_car, break_long_words=False, break_on_hyphens=False)
    grupos = [lineas[i:i + max_lineas] for i in range(0, len(lineas), max_lineas)]
    if len(grupos) > 1:
        # Rebalancea para que los subtítulos tengan longitudes parecidas (evita un último de 2 palabras).
        palabras = texto.split()
        n = len(grupos)
        objetivo = len(texto) / n
        grupos, actual = [], []
        for p in palabras:
            actual.append(p)
            if len(" ".join(actual)) >= objetivo and len(grupos) < n - 1:
                grupos.append(textwrap.wrap(" ".join(actual), max_car))
                actual = []
        grupos.append(textwrap.wrap(" ".join(actual), max_car))
    total = sum(len(" ".join(g)) for g in grupos)
    cues, t = [], inicio
    for g in grupos:
        d = (fin - inicio) * len(" ".join(g)) / total
        cues.append((t, t + d, "\n".join(g)))
        t += d
    return cues


def capitulos(guion: dict, inicios: list[float], total: float) -> tuple[list[tuple[float, str]], list[str]]:
    caps: list[list] = []   # [inicio, título, fin]
    bloque_anterior = None
    for seg, t in zip(guion["segmentos"], inicios):
        if seg["bloque"] != bloque_anterior:
            titulo = seg.get("titulo_capitulo") or TITULOS_POR_DEFECTO.get(seg["bloque"], seg["bloque"])
            caps.append([t, titulo, None])
            bloque_anterior = seg["bloque"]
    for i, c in enumerate(caps):
        c[2] = caps[i + 1][0] if i + 1 < len(caps) else total

    avisos = []
    fusionados = []
    for c in caps:
        if fusionados and c[2] - c[0] < CAPITULO_MIN_S:
            avisos.append(f"capítulo «{c[1]}» dura {c[2] - c[0]:.1f} s (<{CAPITULO_MIN_S} s): se une a «{fusionados[-1][1]}»")
            fusionados[-1][2] = c[2]
        else:
            fusionados.append(c)
    # El primero también puede quedar corto: se une al siguiente conservando el 0:00.
    if len(fusionados) > 1 and fusionados[0][2] - fusionados[0][0] < CAPITULO_MIN_S:
        avisos.append(f"capítulo «{fusionados[0][1]}» dura menos de {CAPITULO_MIN_S} s: se une a «{fusionados[1][1]}»")
        fusionados[1][0] = 0.0
        fusionados.pop(0)
    if len(fusionados) < 3:
        avisos.append(f"solo {len(fusionados)} capítulo(s): YouTube exige al menos 3 (no se mostrarán)")
    return [(c[0], c[1]) for c in fusionados], avisos


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    ajustes = config.cargar("pipeline")["subtitulos"]

    # Misma línea temporal que e6: duración real de cada .wav de segmento, acumulada.
    inicios, t = [], 0.0
    metas = []
    for seg in guion["segmentos"]:
        audio = dir_ep / "build" / "audio" / seg["id"]
        if not audio.with_suffix(".json").exists():
            print(f"[FALLO] falta el audio de {seg['id']}: ejecuta antes e2")
            return comun.FALLO
        metas.append(json.loads(audio.with_suffix(".json").read_text(encoding="utf-8")))
        inicios.append(t)
        t += sf.info(str(audio.with_suffix(".wav"))).duration
    total = t

    cues = []
    for meta, t0 in zip(metas, inicios):
        for f in meta["frases"]:
            cues += partir_frase(f["texto"], t0 + f["inicio"], t0 + f["fin"],
                                 ajustes["max_caracteres_linea"], ajustes["max_lineas"])

    ep = f"ep{int(guion['episodio']):02d}"
    dir_final = config.SALIDA / ep
    dir_final.mkdir(parents=True, exist_ok=True)
    srt = dir_final / f"{guion['slug']}.srt"
    srt.write_text("\n".join(f"{i}\n{tiempo_srt(a)} --> {tiempo_srt(b)}\n{texto}\n"
                             for i, (a, b, texto) in enumerate(cues, 1)), encoding="utf-8")
    largas = [l for _, _, tx in cues for l in tx.splitlines() if len(l) > ajustes["max_caracteres_linea"]]
    print(f"[{'AVISO' if largas else 'OK'}] {srt.relative_to(config.RAIZ)}: {len(cues)} subtítulos"
          + (f", {len(largas)} líneas largas (palabras sin cortar)" if largas else ""))

    caps, avisos = capitulos(guion, inicios, total)
    txt = dir_final / "capitulos.txt"
    txt.write_text("\n".join(f"{tiempo_capitulo(a)} {titulo}" for a, titulo in caps) + "\n", encoding="utf-8")
    for a in avisos:
        print(f"[AVISO] {a}")
    print(f"[OK] {txt.relative_to(config.RAIZ)}: {len(caps)} capítulos")
    for a, titulo in caps:
        print(f"  {tiempo_capitulo(a)} {titulo}")
    return comun.OK


if __name__ == "__main__":
    sys.exit(main())
