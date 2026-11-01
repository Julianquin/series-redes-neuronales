"""Orquestador del pipeline (§10.1).

    python pipeline/produce.py --ep 00                          # pipeline completo
    python pipeline/produce.py --ep 00 --desde e4               # relanzar desde el render
    python pipeline/produce.py --ep 00 --solo e2 --segmento s03 # regenerar una voz
    python pipeline/produce.py --ep 00 --borrador               # render rápido -ql

Cada etapa corre en su propio subproceso (así la VRAM se libera entre etapas),
deja su log en build/logs/<etapa>.log y el orquestador se detiene ante el primer fallo.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import config  # noqa: E402
from pipeline import comun  # noqa: E402

ETAPAS = {
    "e1": ("e1_validar_guion", "Validar guion"),
    "e2": ("e2_tts", "Voz (TTS) y QA de voz"),
    "e3": (None, "Escenas Manim (escenas.py, Claude Code)"),
    "e4": ("e4_render", "Render"),
    "e5": ("e5_hoja_contactos", "Hoja de contactos"),
    "e6": ("e6_ensamblar", "Ensamblado"),
    "e7": ("e7_subtitulos_capitulos", "Subtítulos y capítulos"),
    "e8": ("e8_qa", "QA final"),
    "e9": ("e9_miniatura_metadatos", "Miniatura y metadatos"),
}

# Puntos de control humanos (§12): el orquestador se para aquí para que revises.
CONTROLES = {
    "e2": "Control 1 — ¿has revisado el guion antes de generar la voz?",
    "e6": "Control 2 — revisa build/contactos/ antes de ensamblar.",
}


def comprobar_escenas(dir_ep: Path) -> int:
    """e3 no es un script: escenas.py lo escribe Claude Code. Aquí solo se comprueba que existe
    y que define una clase por cada escena del guion."""
    escenas = dir_ep / "escenas.py"
    if not escenas.exists():
        print(f"[FALLO] falta {escenas}. Pide a Claude Code que lo genere a partir de guion.yaml.")
        return comun.FALLO
    definidas = set(re.findall(r"^class\s+(\w+)\s*\(", escenas.read_text(encoding="utf-8"), re.M))
    faltan = [s["escena"] for s in comun.cargar_guion(dir_ep)["segmentos"] if s["escena"] not in definidas]
    if faltan:
        print(f"[FALLO] escenas.py no define: {', '.join(faltan)}")
        return comun.FALLO
    print(f"[OK] escenas.py define las {len(definidas)} escenas del guion")
    return comun.OK


def ejecutar(etapa: str, dir_ep: Path, args: argparse.Namespace) -> int:
    modulo, nombre = ETAPAS[etapa]
    print(f"\n=== {etapa} · {nombre} ===", flush=True)
    if modulo is None:
        return comprobar_escenas(dir_ep)

    # La etapa de voz puede vivir en otro venv (Qwen3-TTS pide otra versión de transformers).
    interprete = sys.executable
    if etapa == "e2":
        propio = config.cargar("voz").get("python")
        if propio:
            interprete = str(config.RAIZ / propio)
            if not Path(interprete).exists():
                print(f"[FALLO] falta el intérprete de voz {propio} (ver env/setup_qwen.sh)")
                return comun.FALLO
    cmd = [interprete, "-m", f"pipeline.{modulo}", "--ep", dir_ep.name]
    if args.segmento:
        cmd += ["--segmento", args.segmento]
    if args.borrador:
        cmd.append("--borrador")
    if args.simulado:
        cmd.append("--simulado")

    log = comun.dir_build(dir_ep, "logs") / f"{etapa}.log"
    inicio = time.monotonic()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=config.RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for linea in proc.stdout:
            print(linea, end="")
            f.write(linea)
        codigo = proc.wait()
    print(f"--- {etapa} terminó con código {codigo} en {time.monotonic() - inicio:.1f} s (log: {log.relative_to(config.RAIZ)})")
    return codigo


def main() -> int:
    p = argparse.ArgumentParser(description="Orquestador del pipeline de la serie")
    p.add_argument("--ep", required=True)
    grupo = p.add_mutually_exclusive_group()
    grupo.add_argument("--desde", choices=ETAPAS, help="empezar en esta etapa")
    grupo.add_argument("--solo", choices=ETAPAS, help="ejecutar solo esta etapa")
    p.add_argument("--hasta", choices=ETAPAS, help="terminar en esta etapa")
    p.add_argument("--segmento")
    p.add_argument("--borrador", action="store_true")
    p.add_argument("--sin-pausas", action="store_true", help="no detenerse en los puntos de control")
    p.add_argument("--simulado", action="store_true", help="sin modelos de IA, para probar el pipeline")
    args = p.parse_args()

    dir_ep = config.dir_episodio(args.ep)
    orden = list(ETAPAS)
    if args.solo:
        etapas = [args.solo]
    else:
        desde = orden.index(args.desde) if args.desde else 0
        hasta = orden.index(args.hasta) + 1 if args.hasta else len(orden)
        etapas = orden[desde:hasta]

    print(f"Episodio: {dir_ep.name} · etapas: {' → '.join(etapas)}")
    for i, etapa in enumerate(etapas):
        # Los controles solo paran cuando se llega a la etapa desde una anterior en la misma ejecución.
        if i > 0 and etapa in CONTROLES and not args.sin_pausas:
            print(f"\n⏸  {CONTROLES[etapa]}\n   Cuando esté revisado: python pipeline/produce.py --ep {args.ep} --desde {etapa}")
            return comun.OK
        codigo = ejecutar(etapa, dir_ep, args)
        if codigo == comun.PENDIENTE:
            print(f"\n⏹  Parado: {etapa} aún no está implementada.")
            return codigo
        if codigo != comun.OK:
            print(f"\n✖  Parado: {etapa} falló. Revisa el log y relanza con --desde {etapa}.")
            return codigo
    print("\n✔  Etapas completadas.")
    return comun.OK


if __name__ == "__main__":
    sys.exit(main())
