"""Utilidades compartidas por las etapas del pipeline.

Cada etapa es un script independiente que se ejecuta como subproceso
(`python -m pipeline.e2_tts --ep 00`) y comunica su resultado por el código
de salida.
"""
import argparse
import sys
from pathlib import Path

import yaml

# Códigos de salida acordados entre las etapas y el orquestador.
OK = 0
FALLO = 1
PENDIENTE = 3   # etapa todavía sin implementar

BLOQUES = ("gancho", "limitacion", "intuicion", "matematicas", "demo", "cuando_usar", "puente")


def argumentos(descripcion: str, extra=None) -> argparse.Namespace:
    """Argumentos comunes a todas las etapas; `extra(parser)` añade los propios de una etapa."""
    p = argparse.ArgumentParser(description=descripcion, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ep", required=True, help="número de episodio (00) o slug (ep00_mlp)")
    p.add_argument("--segmento", help="limitar la etapa a un segmento (p. ej. s03)")
    p.add_argument("--borrador", action="store_true", help="calidad de borrador (-ql)")
    p.add_argument("--simulado", action="store_true", help="sin modelos: voz y transcripción falsas (pruebas)")
    if extra:
        extra(p)
    return p.parse_args()


def cargar_guion(dir_ep: Path) -> dict:
    with open(dir_ep / "guion.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dir_build(dir_ep: Path, *partes: str) -> Path:
    ruta = dir_ep.joinpath("build", *partes)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


FPS = {"ql": 15, "qm": 30, "qh": 60, "qp": 60, "qk": 60}


def calidad(args: argparse.Namespace) -> str:
    """Calidad de render de esta ejecución: borrador (-ql) o final (-qh), según config/pipeline.yaml."""
    from lib import config

    ajustes = config.cargar("pipeline")["render"]
    return ajustes["calidad_borrador"] if args.borrador else ajustes["calidad_final"]


def ffprobe(ruta: Path, entrada: str, flujo: str | None = None) -> str:
    """Un campo de ffprobe, p. ej. ffprobe(v, "stream=duration", "v:0") o ffprobe(v, "format=duration")."""
    import subprocess

    cmd = ["ffprobe", "-v", "error", "-show_entries", entrada, "-of", "csv=p=0", str(ruta)]
    if flujo:
        cmd[3:3] = ["-select_streams", flujo]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def duracion_video(ruta: Path) -> float:
    """Duración de la pista de vídeo (no del contenedor, que toma la del audio si es más larga):
    un desfase de un fotograma por segmento se acumula al concatenar."""
    return float(ffprobe(ruta, "stream=duration", "v:0"))


def ffmpeg(*args: str, log: Path | None = None) -> str:
    """Ejecuta ffmpeg; si falla, lanza RuntimeError con el final de su salida de error."""
    import subprocess

    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True)
    if log:
        log.write_text(r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg falló:\n" + "\n".join(r.stderr.strip().splitlines()[-8:]))
    return r.stderr


def sonoridad(ruta: Path) -> tuple[float, float]:
    """(sonoridad integrada en LUFS, pico real en dBTP) medidos con ebur128."""
    import re

    salida = ffmpeg("-nostats", "-i", str(ruta), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-")
    resumen = salida[salida.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s*(-?[\d.]+) LUFS", resumen).group(1))
    pico = float(re.search(r"Peak:\s*(-?[\d.inf]+) dBFS", resumen).group(1))
    return lufs, pico


def pendiente(nombre: str) -> None:
    """Salida estándar de una etapa que aún no está implementada."""
    print(f"[PENDIENTE] {nombre}: etapa sin implementar todavía")
    sys.exit(PENDIENTE)

