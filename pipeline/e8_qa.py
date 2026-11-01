"""Etapa 8 — QA final del vídeo ensamblado (§10.10).

Comprueba el vídeo de salida/epXX/ y escribe salida/epXX/informe_qa[_borrador].md con OK, AVISO o
FALLO por comprobación. Cualquier FALLO detiene el pipeline.

    python -m pipeline.e8_qa --ep 99 [--borrador]
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from lib import config
from pipeline import comun

OK, AVISO, FALLO = "OK", "AVISO", "FALLO"


def detectar(video: Path, filtro: str, audio: bool) -> str:
    flujo = ["-vn"] if audio else ["-an"]
    campo = "-af" if audio else "-vf"
    return comun.ffmpeg("-i", str(video), *flujo, campo, filtro, "-f", "null", "-")


def intervalos(salida: str, inicio: str, fin: str) -> list[tuple[float, float]]:
    """Pares (inicio, duración) de blackdetect / freezedetect / silencedetect."""
    ini = [float(x) for x in re.findall(rf"{inicio}:\s*(-?[\d.]+)", salida)]
    dur = [float(x) for x in re.findall(rf"{fin}:\s*([\d.]+)", salida)]
    return list(zip(ini, dur))


def comprobaciones(dir_ep: Path, guion: dict, video: Path, borrador: bool) -> list[tuple[str, str, str]]:
    qa = config.cargar("pipeline")["qa_final"]
    au = config.cargar("pipeline")["audio"]
    r: list[tuple[str, str, str]] = []

    # Voz: estado del QA de cada frase y método de las marcas.
    metas = [json.loads((dir_ep / "build" / "audio" / f"{s['id']}.json").read_text(encoding="utf-8"))
             for s in guion["segmentos"]]
    revisar = [m["segmento"] for m in metas if m["qa"]["estado"] == "revisar"]
    simulados = [m["segmento"] for m in metas if m["qa"]["estado"] == "simulado"]
    if revisar:
        r.append(("Voz", FALLO, f"segmentos con frases sin aprobar: {', '.join(revisar)}"))
    elif simulados:
        r.append(("Voz", AVISO if borrador else FALLO, f"voz SIMULADA en {', '.join(simulados)}: nunca publicar"))
    else:
        r.append(("Voz", OK, f"{len(metas)} segmentos con QA de voz aprobado"))
    estimadas = [f"{m['segmento']}:{e}" for m in metas for e, metodo in m.get("marcas_metodo", {}).items()
                 if metodo == "estimada"]
    r.append(("Marcas", AVISO if estimadas else OK,
              f"estimadas (Whisper no las oyó): {', '.join(estimadas)}" if estimadas else "todas localizadas por Whisper"))

    # Avisos del render (animaciones tarde, finales sin tiempo para verse).
    calidad = comun.calidad(type("A", (), {"borrador": borrador})())
    avisos_render = [l for s in guion["segmentos"]
                     for f in [dir_ep / "build" / "render" / calidad / f"{s['id']}.avisos"] if f.exists()
                     for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    r.append(("Render", AVISO if avisos_render else OK,
              "; ".join(a.replace("[AVISO] ", "") for a in avisos_render) or "sin avisos de sincronización"))

    # Duraciones.
    voz = sum(m["duracion"] for m in metas)
    d_video = comun.duracion_video(video)
    fotograma = 1 / comun.FPS.get(calidad, 60)
    r.append(("Vídeo frente a voz", OK if abs(d_video - voz) <= fotograma + 0.01 else FALLO,
              f"vídeo {d_video:.3f} s · voz {voz:.3f} s · diferencia {d_video - voz:+.3f} s"))
    objetivo = guion["duracion_objetivo_min"] * 60
    desvio = (d_video - objetivo) / objetivo
    r.append(("Duración total", OK if abs(desvio) <= qa["tolerancia_duracion_total"] else AVISO,
              f"{d_video / 60:.2f} min frente a {guion['duracion_objetivo_min']} min objetivo ({desvio:+.0%})"))

    # Imagen: negros (el fondo #111318 no cuenta como negro con pix_th=0.03) y congelados.
    negros = [(a, d) for a, d in intervalos(detectar(video, f"blackdetect=d={qa['negro_max_s']}:pix_th=0.03", False),
                                            "black_start", "black_duration")]
    r.append(("Pantallas negras", FALLO if negros else OK,
              ", ".join(f"{a:.1f} s ({d:.1f} s)" for a, d in negros) or f"ninguna de más de {qa['negro_max_s']} s"))
    congelados = intervalos(detectar(video, f"freezedetect=n=-60dB:d={qa['congelado_aviso_s']}", False),
                            "freeze_start", "freeze_duration")
    r.append(("Imagen congelada", AVISO if congelados else OK,
              ", ".join(f"{a:.1f} s ({d:.1f} s)" for a, d in congelados)
              or f"ninguna de más de {qa['congelado_aviso_s']} s"))

    # Audio: silencios y sonoridad.
    silencios = intervalos(detectar(video, f"silencedetect=n=-45dB:d={qa['silencio_aviso_s']}", True),
                           "silence_start", "silence_duration")
    r.append(("Silencios largos", AVISO if silencios else OK,
              ", ".join(f"{a:.1f} s ({d:.1f} s)" for a, d in silencios)
              or f"ninguno de más de {qa['silencio_aviso_s']} s"))
    lufs, pico = comun.sonoridad(video)
    r.append(("Sonoridad", OK if abs(lufs - au["lufs_objetivo"]) <= 1 else FALLO,
              f"{lufs:.1f} LUFS (objetivo {au['lufs_objetivo']} ± 1)"))
    r.append(("Pico real", OK if pico <= au["true_peak"] + 0.5 else AVISO, f"{pico:.1f} dBTP (máx. {au['true_peak']})"))

    # Subtítulos y capítulos (e7).
    dir_final = video.parent
    srt = dir_final / f"{guion['slug']}.srt"
    if srt.exists():
        fines = re.findall(r"--> (\d\d):(\d\d):(\d\d),(\d{3})", srt.read_text(encoding="utf-8"))
        ultimo = max(int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000 for h, m, s, ms in fines) if fines else 0
        r.append(("Subtítulos", OK if ultimo <= d_video + 0.05 else FALLO,
                  f"{len(fines)} subtítulos; el último termina en {ultimo:.2f} s"))
    else:
        r.append(("Subtítulos", FALLO, "falta el .srt: ejecuta e7"))
    caps = dir_final / "capitulos.txt"
    n_caps = len(caps.read_text(encoding="utf-8").splitlines()) if caps.exists() else 0
    r.append(("Capítulos", OK if n_caps >= 3 else AVISO,
              f"{n_caps} capítulos" + ("" if n_caps >= 3 else " (YouTube exige al menos 3)")))
    return r


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    dir_final = config.SALIDA / f"ep{int(guion['episodio']):02d}"
    sufijo = "_borrador" if args.borrador else ""
    video = dir_final / f"{guion['slug']}{sufijo}.mp4"
    if not video.exists():
        print(f"[FALLO] no existe {video.relative_to(config.RAIZ)}: ejecuta antes e6")
        return comun.FALLO

    resultados = comprobaciones(dir_ep, guion, video, args.borrador)
    for nombre, estado, detalle in resultados:
        print(f"[{estado}] {nombre}: {detalle}")

    iconos = {OK: "✅", AVISO: "⚠️", FALLO: "❌"}
    n = {e: sum(1 for _, x, _ in resultados if x == e) for e in (OK, AVISO, FALLO)}
    informe = dir_final / f"informe_qa{sufijo}.md"
    informe.write_text(
        f"# Informe de QA — {guion['titulo']}\n\n"
        f"- Vídeo: `{video.relative_to(config.RAIZ)}`\n"
        f"- Generado: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"- Resultado: {n[OK]} OK · {n[AVISO]} avisos · {n[FALLO]} fallos\n\n"
        "| Comprobación | Estado | Detalle |\n|---|---|---|\n"
        + "".join(f"| {nombre} | {iconos[e]} {e} | {detalle} |\n" for nombre, e, detalle in resultados),
        encoding="utf-8")
    print(f"Informe: {informe.relative_to(config.RAIZ)} · {n[OK]} OK · {n[AVISO]} avisos · {n[FALLO]} fallos")
    return comun.FALLO if n[FALLO] else comun.OK


if __name__ == "__main__":
    sys.exit(main())
