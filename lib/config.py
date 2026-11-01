"""Rutas del repositorio y carga de la configuración YAML."""
from functools import cache
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = RAIZ / "config"
EPISODIOS = RAIZ / "episodios"
SALIDA = RAIZ / "salida"


@cache
def cargar(nombre: str) -> dict:
    """Carga config/<nombre>.yaml (se cachea: la configuración no cambia durante un proceso)."""
    with open(CONFIG / f"{nombre}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dir_episodio(ep: str | int) -> Path:
    """Resuelve '00', 0 o 'ep00_mlp' al directorio episodios/ep00_<slug>."""
    if isinstance(ep, str) and ep.startswith("ep"):
        candidatos = [EPISODIOS / ep]
    else:
        candidatos = sorted(EPISODIOS.glob(f"ep{int(ep):02d}_*"))
    candidatos = [c for c in candidatos if c.is_dir()]
    if len(candidatos) != 1:
        raise FileNotFoundError(f"No encuentro un único directorio para el episodio {ep!r} en {EPISODIOS}")
    return candidatos[0]
