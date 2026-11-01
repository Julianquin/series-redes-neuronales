"""Etapa 4 — Render de un .mp4 por segmento, en paralelo y con caché (§10.6).

Cada segmento se renderiza con `manim` en su propio subproceso y su propia carpeta de trabajo
(build/media/<seg>/): así no chocan al compilar LaTeX. El resultado va a
build/render/<calidad>/<seg>.mp4, con el audio de la voz ya incluido por EscenaNarrada.

Caché: si no cambian escenas.py, lib/, el estilo, el audio del segmento ni la calidad, no se
vuelve a renderizar. Al terminar comprueba que cada vídeo dura lo mismo que su audio.

    python -m pipeline.e4_render --ep 99 [--borrador] [--segmento s02]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from lib import config
from pipeline import comun


def huella(dir_ep: Path, seg: dict, calidad: str) -> str:
    """Todo lo que influye en el vídeo de un segmento."""
    h = hashlib.sha256()
    h.update(f"{calidad}|{seg['escena']}|manim {version('manim')}".encode())
    fuentes = [dir_ep / "escenas.py", config.CONFIG / "estilo.yaml", *sorted((config.RAIZ / "lib").rglob("*.py"))]
    audio = dir_ep / "build" / "audio"
    fuentes += [audio / f"{seg['id']}.json", audio / f"{seg['id']}.wav"]
    for f in fuentes:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def procesos_paralelos(n_segmentos: int) -> int:
    ajuste = config.cargar("pipeline")["render"]["procesos_paralelos"]
    maximo = max(1, (os.cpu_count() or 2) - 1) if ajuste == "auto" else int(ajuste)
    return max(1, min(maximo, n_segmentos))


def renderizar(dir_ep: Path, seg: dict, calidad: str, dir_salida: Path) -> dict:
    sid = seg["id"]
    salida = dir_salida / f"{sid}.mp4"
    marca_huella = salida.with_suffix(".huella")
    clave = huella(dir_ep, seg, calidad)
    ruta_avisos = salida.with_suffix(".avisos")
    if salida.exists() and marca_huella.exists() and marca_huella.read_text() == clave:
        avisos = ruta_avisos.read_text(encoding="utf-8").splitlines() if ruta_avisos.exists() else []
        return {"segmento": sid, "estado": "caché", "avisos": avisos}

    media = dir_ep / "build" / "media" / sid
    log = comun.dir_build(dir_ep, "logs") / f"e4_{sid}.log"
    cmd = [str(Path(sys.executable).with_name("manim")), f"-{calidad}", "--media_dir", str(media),
           "-o", sid, "--progress_bar", "none", str(dir_ep / "escenas.py"), seg["escena"]]
    r = subprocess.run(cmd, cwd=config.RAIZ, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr, encoding="utf-8")
    avisos = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "[AVISO]" in l]
    if r.returncode != 0:
        return {"segmento": sid, "estado": "fallo", "avisos": avisos,
                "error": f"manim terminó con código {r.returncode} (log: {log.relative_to(config.RAIZ)})"}

    generados = sorted(media.glob(f"videos/**/{sid}.mp4"), key=lambda p: p.stat().st_mtime)
    if not generados:
        return {"segmento": sid, "estado": "fallo", "avisos": avisos, "error": "manim no dejó el .mp4 esperado"}
    shutil.copy2(generados[-1], salida)
    marca_huella.write_text(clave)
    ruta_avisos.write_text("\n".join(avisos), encoding="utf-8")   # se muestran también desde la caché
    return {"segmento": sid, "estado": "renderizado", "avisos": avisos}


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    calidad = comun.calidad(args)
    dir_salida = comun.dir_build(dir_ep, "render", calidad)
    tolerancia = config.cargar("pipeline")["qa_final"]["tolerancia_duracion_segmento_s"]

    segmentos = [s for s in guion["segmentos"] if not args.segmento or s["id"] == args.segmento]
    faltan = [s["id"] for s in segmentos if not (dir_ep / "build" / "audio" / f"{s['id']}.json").exists()]
    if faltan:
        print(f"[FALLO] falta el audio de {', '.join(faltan)}: ejecuta antes la etapa e2")
        return comun.FALLO

    n = procesos_paralelos(len(segmentos))
    print(f"Render -{calidad} de {len(segmentos)} segmentos con {n} procesos → {dir_salida.relative_to(config.RAIZ)}",
          flush=True)
    fallos = 0
    with ThreadPoolExecutor(max_workers=n) as pool:
        for res in pool.map(lambda s: renderizar(dir_ep, s, calidad, dir_salida), segmentos):
            sid = res["segmento"]
            if res["estado"] == "fallo":
                fallos += 1
                print(f"[FALLO] {sid}: {res['error']}", flush=True)
                continue
            audio = json.loads((dir_ep / "build" / "audio" / f"{sid}.json").read_text(encoding="utf-8"))
            d_video = comun.duracion_video(dir_salida / f"{sid}.mp4")
            desfase = d_video - audio["duracion"]
            # Quedarse corto es inofensivo: e6 rellena cada segmento hasta la duración de su voz.
            # Manim redondea cada animación a fotogramas completos, así que una escena con muchas
            # animaciones se queda corta hasta ~4 fotogramas (en borrador, a 15 fps, 0,27 s).
            # Pasarse sí importa: la imagen invadiría el segmento siguiente.
            fotograma = 1 / comun.FPS.get(calidad, 60)
            ok = -(tolerancia + 4 * fotograma) <= desfase <= fotograma + 0.01
            print(f"[{'OK' if ok else 'FALLO'}] {sid} ({res['estado']}): vídeo {d_video:.2f} s · audio "
                  f"{audio['duracion']:.2f} s · desfase {desfase:+.2f} s"
                  + ("" if ok or desfase < 0 else " → la escena dura más que la voz: acorta la última animación"),
                  flush=True)
            for aviso in res["avisos"]:
                print(f"  {aviso}", flush=True)
            fallos += not ok
    return comun.FALLO if fallos else comun.OK


if __name__ == "__main__":
    sys.exit(main())
