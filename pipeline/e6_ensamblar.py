"""Etapa 6 — Ensamblado: vídeo continuo, voz, música con ducking, sonoridad y NVENC (§10.8).

La voz manda: se concatenan los .wav originales de cada segmento (48 kHz, sin recomprimir) y cada
vídeo se ajusta a un número exacto de fotogramas calculado sobre el tiempo acumulado, así que el
desfase entre imagen y voz nunca supera un fotograma por largo que sea el episodio.

Audio: voz (+ música en bucle que baja sola cuando hay voz y se funde al final) → ganancia y
limitador de picos hasta config/pipeline.yaml → audio.lufs_objetivo (−14 LUFS, referencia de YouTube).

Salida: salida/epXX/<slug>.mp4 (o <slug>_borrador.mp4 con --borrador).

    python -m pipeline.e6_ensamblar --ep 99 [--borrador]
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from lib import config
from pipeline import comun

SR = 48000


def unir_voz(dir_ep: Path, segmentos: list[dict], destino: Path) -> list[float]:
    """Concatena los .wav de los segmentos y devuelve la duración de cada uno."""
    partes, duraciones = [], []
    for seg in segmentos:
        wav, sr = sf.read(dir_ep / "build" / "audio" / f"{seg['id']}.wav", dtype="float32")
        if sr != SR:
            raise ValueError(f"{seg['id']}.wav está a {sr} Hz; se esperaban {SR}")
        partes.append(wav)
        duraciones.append(len(wav) / SR)
    sf.write(destino, np.concatenate(partes), SR, subtype="PCM_24")
    return duraciones


def fotogramas_por_segmento(duraciones: list[float], fps: int) -> list[int]:
    """Fotogramas de cada segmento redondeando sobre el tiempo acumulado (error < 1 fotograma)."""
    fin = np.round(np.cumsum(duraciones) * fps).astype(int)
    return list(np.diff(np.concatenate([[0], fin])))


def tramos_musica(guion: dict, inicios: dict[str, float], total: float) -> list[dict]:
    """Normaliza el campo `musica` del guion a una lista de tramos con su posición en el vídeo.

    Forma simple:   musica: assets/musica/pista.mp3          (desde el principio, en bucle)
    Por tramos:     musica:
                      - pista: assets/musica/a.mp3
                        desde: s01          # empieza con este segmento (por defecto, el primero)
                        entrada: s02        # la transición de la pista cae al empezar s02 (opcional)
                        transicion: auto    # segundo de la pista; auto = primera subida de energía
                      - pista: assets/musica/b.mp3
                        desde: s05          # fundido cruzado con la anterior al empezar s05
    """
    from lib import musica

    spec = guion.get("musica")
    if not spec:
        return []
    if isinstance(spec, str):
        spec = [{"pista": spec}]
    primero = guion["segmentos"][0]["id"]
    tramos = []
    for i, t in enumerate(spec):
        ruta = config.RAIZ / t["pista"]
        if not ruta.exists():
            raise FileNotFoundError(f"no existe la música {t['pista']}")
        analisis = musica.analizar(ruta)
        desde = t.get("desde", primero)
        inicio = inicios[desde]
        # Instante de la pista que suena en `inicio`: sin alinear, su principio; alineado, el que hace
        # que `transicion` caiga en el inicio del segmento `entrada`.
        pista_en_inicio = 0.0
        if t.get("entrada"):
            transicion = t.get("transicion", "auto")
            if transicion == "auto":
                transicion = musica.primera_subida(analisis)
                if transicion is None:
                    raise ValueError(f"{t['pista']}: no tiene una subida clara; indica `transicion` en segundos")
            pista_en_inicio = float(transicion) - (inicios[t["entrada"]] - inicio)
        tramos.append({"pista": ruta, "nombre": ruta.stem, "inicio": inicio, "pista_en_inicio": pista_en_inicio,
                       "lufs": analisis["lufs"], "duracion_pista": analisis["duracion"],
                       "transicion": t.get("entrada") and (float(pista_en_inicio + inicios[t["entrada"]] - inicio))})
    tramos.sort(key=lambda t: t["inicio"])
    for i, t in enumerate(tramos):
        t["fin"] = tramos[i + 1]["inicio"] if i + 1 < len(tramos) else total
    return tramos


def mezclar_musica(voz: Path, tramos: list[dict], duracion: float, destino: Path, log: Path) -> list[str]:
    """Tramos de música bajo la voz: cada uno igualado en sonoridad a musica_bajo_voz_lu por debajo
    de la voz, con fundido cruzado entre tramos y de salida al final; sidechaincompress la baja
    mientras hablas. Devuelve avisos (p. ej. un tramo más largo que su pista, que se repetirá)."""
    a = config.cargar("pipeline")["audio"]
    fundido = a["fundido_musica_s"]
    d = a["ducking"]
    voz_lufs, _ = comun.sonoridad(voz)
    objetivo = voz_lufs - a["musica_bajo_voz_lu"]
    entradas, cadenas, avisos = ["-i", str(voz)], [], []
    for i, t in enumerate(tramos):
        # El fundido cruzado se reparte a ambos lados del cambio de tramo.
        t0 = t["inicio"] - (fundido / 2 if i > 0 else 0.0)
        t1 = t["fin"] + (fundido / 2 if i + 1 < len(tramos) else 0.0)
        pos = t["pista_en_inicio"] - (t["inicio"] - t0)       # instante de la pista en t0
        retraso = max(0.0, -pos)                               # si la pista aún no ha empezado en t0
        pos = max(0.0, pos)
        dur = t1 - t0 - retraso
        if pos + dur > t["duracion_pista"]:
            avisos.append(f"{t['nombre']}: el tramo necesita {pos + dur:.0f} s y la pista dura "
                          f"{t['duracion_pista']:.0f} s; se repetirá desde el principio")
        entrada_suave = fundido if i > 0 else (0.5 if pos > 0 else 0.0)
        cadena = (f"[{i + 1}:a]aresample={SR},aformat=channel_layouts=stereo,"
                  f"atrim=start={pos:.3f}:duration={dur:.3f},asetpts=PTS-STARTPTS,"
                  f"volume={objetivo - t['lufs']:.2f}dB")
        if entrada_suave:
            cadena += f",afade=t=in:d={entrada_suave:.2f}"
        cadena += f",afade=t=out:st={max(0.0, dur - fundido):.3f}:d={fundido:.2f}"
        cadena += f",adelay={int(round((t0 + retraso) * 1000))}:all=1[m{i}]"
        cadenas.append(cadena)
        entradas += ["-stream_loop", "-1", "-i", str(t["pista"])]
    n = len(tramos)
    mezcla_mus = "".join(f"[m{i}]" for i in range(n)) + (
        f"amix=inputs={n}:normalize=0:duration=longest[mus]" if n > 1 else "anull[mus]")
    filtro = ";".join(cadenas + [
        mezcla_mus,
        "[0:a]aformat=channel_layouts=stereo,asplit=2[voz][sc]",
        f"[mus][sc]sidechaincompress=threshold={d['umbral']}:ratio={d['ratio']}:attack={d['ataque_ms']}:"
        f"release={d['liberacion_ms']},asplit=2[musd][stem]",
        "[voz][musd]amix=inputs=2:duration=first:normalize=0[a]",
    ])
    # Se guarda también la música sola ya atenuada, para medirla y escucharla aparte.
    stem = destino.with_name("musica.wav")
    comun.ffmpeg(*entradas, "-filter_complex", filtro, "-map", "[a]", "-t", f"{duracion:.3f}",
                 "-ar", str(SR), "-c:a", "pcm_s24le", str(destino),
                 "-map", "[stem]", "-t", f"{duracion:.3f}", "-ar", str(SR), "-c:a", "pcm_s24le", str(stem), log=log)
    return avisos


def normalizar(entrada: Path, destino: Path, log: Path) -> dict:
    """Lleva el audio a audio.lufs_objetivo sin pasar de audio.true_peak.

    loudnorm no sirve aquí: los picos de la voz (−1 dBFS) le impiden subir +6 dB en modo lineal,
    cambia a modo dinámico y se queda ~1 LU corto. En su lugar: ganancia + limitador de picos,
    se mide el resultado y se corrige la ganancia hasta quedar a ±0,1 LU (máx. 4 pasadas)."""
    a = config.cargar("pipeline")["audio"]
    objetivo = a["lufs_objetivo"]
    # El limitador trabaja con picos de muestra; el pico real (y el AAC) suben ~0,5–1 dB más.
    limite = 10 ** ((a["true_peak"] - 1.0) / 20)
    inicial, pico_inicial = comun.sonoridad(entrada)
    ganancia = objetivo - inicial
    for _ in range(4):
        # Estéreo ANTES del limitador: al duplicar la voz mono, ffmpeg baja cada canal 3 dB; si se
        # hace después, el limitador recorta picos que luego bajarían solos y se pierde ~1,4 LU.
        filtro = (f"aformat=channel_layouts=stereo,volume={ganancia:.2f}dB,"
                  f"alimiter=limit={limite:.4f}:attack=5:release=50:level=false,aresample={SR}")
        comun.ffmpeg("-i", str(entrada), "-af", filtro, "-c:a", "pcm_s24le", str(destino), log=log)
        final, pico = comun.sonoridad(destino)
        if abs(objetivo - final) <= 0.1:
            break
        ganancia += objetivo - final
    return {"inicial_lufs": inicial, "inicial_pico": pico_inicial, "final_lufs": final, "final_pico": pico,
            "ganancia_db": round(ganancia, 2)}


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    calidad = comun.calidad(args)
    fps = comun.FPS.get(calidad, 60)
    dir_render = dir_ep / "build" / "render" / calidad
    trabajo = comun.dir_build(dir_ep, "ensamblado", calidad)
    logs = comun.dir_build(dir_ep, "logs")
    segmentos = guion["segmentos"]   # el ensamblado siempre es del episodio completo

    faltan = [s["id"] for s in segmentos if not (dir_render / f"{s['id']}.mp4").exists()]
    if faltan:
        print(f"[FALLO] faltan vídeos de {', '.join(faltan)}: ejecuta antes e4{' --borrador' if args.borrador else ''}")
        return comun.FALLO

    # 1. Voz continua: fija la duración de todo.
    voz = trabajo / "voz.wav"
    duraciones = unir_voz(dir_ep, segmentos, voz)
    total = sum(duraciones)
    print(f"Voz: {len(segmentos)} segmentos, {total:.2f} s", flush=True)

    # 2. Música (opcional) y sonoridad.
    mezcla = voz
    inicios = dict(zip([s["id"] for s in segmentos], np.concatenate([[0.0], np.cumsum(duraciones)[:-1]])))
    try:
        tramos = tramos_musica(guion, inicios, total)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[FALLO] música: {e}")
        return comun.FALLO
    if tramos:
        mezcla = trabajo / "mezcla.wav"
        for aviso in mezclar_musica(voz, tramos, total, mezcla, logs / "e6_musica.log"):
            print(f"[AVISO] {aviso}", flush=True)
        for t in tramos:
            alineada = (f", su transición ({t['transicion']:.1f} s de la pista) cae en el vídeo en "
                        f"{t['inicio'] + t['transicion'] - t['pista_en_inicio']:.2f} s") if t["transicion"] else ""
            print(f"Música: {t['nombre']} de {t['inicio']:.2f} a {t['fin']:.2f} s{alineada}", flush=True)
    else:
        print("Música: ninguna (guion.musica vacío)", flush=True)
    audio_final = trabajo / "audio_final.wav"
    medido = normalizar(mezcla, audio_final, logs / "e6_sonoridad.log")
    print(f"Sonoridad: {medido['inicial_lufs']:.1f} → {medido['final_lufs']:.1f} LUFS "
          f"(pico {medido['final_pico']:.1f} dBTP, ganancia {medido['ganancia_db']:+.1f} dB)", flush=True)

    # 3. Vídeo: cada segmento a su número exacto de fotogramas (rellenando con el último si falta).
    n_fotogramas = fotogramas_por_segmento(duraciones, fps)
    entradas, filtros = [], []
    for i, (seg, n) in enumerate(zip(segmentos, n_fotogramas)):
        entradas += ["-i", str(dir_render / f"{seg['id']}.mp4")]
        filtros.append(f"[{i}:v]fps={fps},tpad=stop_mode=clone:stop_duration=5,trim=end_frame={n},"
                       f"setpts=PTS-STARTPTS[v{i}]")
    filtros.append("".join(f"[v{i}]" for i in range(len(segmentos))) + f"concat=n={len(segmentos)}:v=1:a=0[v]")
    entradas += ["-i", str(audio_final)]

    ep = f"ep{int(guion['episodio']):02d}"
    dir_final = config.SALIDA / ep
    dir_final.mkdir(parents=True, exist_ok=True)
    final = dir_final / f"{guion['slug']}{'_borrador' if args.borrador else ''}.mp4"
    comun.ffmpeg(*entradas, "-filter_complex", ";".join(filtros), "-map", "[v]", "-map", f"{len(segmentos)}:a",
                 "-c:v", "h264_nvenc", "-preset", "p7", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-movflags", "+faststart", str(final),
                 log=logs / "e6_video.log")

    d_video = comun.duracion_video(final)
    d_audio = float(comun.ffprobe(final, "stream=duration", "a:0"))
    ok = abs(d_video - total) <= 1 / fps + 0.01
    print(f"[{'OK' if ok else 'FALLO'}] {final.relative_to(config.RAIZ)}: vídeo {d_video:.3f} s · audio "
          f"{d_audio:.3f} s · voz {total:.3f} s", flush=True)
    (trabajo / "ensamblado.json").write_text(json.dumps(
        {"final": str(final.relative_to(config.RAIZ)), "fps": fps, "duraciones": duraciones,
         "fotogramas": [int(n) for n in n_fotogramas], "sonoridad": medido}, indent=1), encoding="utf-8")
    return comun.OK if ok else comun.FALLO


if __name__ == "__main__":
    sys.exit(main())
