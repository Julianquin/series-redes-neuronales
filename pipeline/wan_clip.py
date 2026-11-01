"""Genera clips de vídeo con Wan 2.2 (TI2V-5B), a partir de un texto o de una imagen.

Herramienta suelta, todavía no una etapa del pipeline: sirve para probar si el vídeo generativo
aporta algo a la serie antes de montarlo (ver "Mejoras pendientes" en README.md).

Se ejecuta con el intérprete de los modelos generativos, el mismo de Qwen (ver env/setup_qwen.sh):

    .venv-qwen/bin/python pipeline/wan_clip.py "una cuadrícula azul que se pliega, minimalista" \\
        --salida build/broll/prueba.mp4 [--imagen primer_fotograma.png] [--segundos 3]

Para varios clips, el modelo se carga una sola vez (tarda unos minutos), igual que el trabajo.json
de notebooks/generar_qwen.py:

    .venv-qwen/bin/python pipeline/wan_clip.py --lote trabajo.json

    {"salida": "build/broll", "clips": [{"id": "invernadero", "prompt": "...", "segundos": 3}, ...]}

Escribe <salida>/<id>.mp4 y, al terminar, <salida>/resultado.json. Los clips que ya existen se
saltan, para poder retomar un lote interrumpido; --rehacer los regenera.

Para decidir si el vídeo generativo aporta algo, primero la versión barata (~2 min en vez de ~20):

    ... --segundos 2 --pasos 25 --alto 480 --ancho 832

Notas de la documentación de Wan 2.2 TI2V-5B: 704x1280, 24 fps, 50 pasos, guidance 5.0, y unos
9 minutos por cada 5 s de vídeo en una GPU de consumo.

Los pesos del repositorio de diffusers vienen en float32 y ocupan 32 GB: cargarlos necesita unos
22 GB de RAM (no de VRAM). Con WSL limitado a 24 GB el kernel mata el proceso; ver .wslconfig.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # Xet se colgó dos veces en esta máquina

import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline  # noqa: E402
from diffusers.utils import export_to_video, load_image  # noqa: E402

MODELO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FPS = 24
ALTO, ANCHO = 704, 1280
MULTIPLO = 32   # el VAE comprime 16x y el transformer agrupa de dos en dos

NEGATIVO = ("texto, letras, subtítulos, marca de agua, logotipo, personas, manos, caras, "
            "deformaciones, parpadeo, colores chillones, saturación excesiva, dibujo animado, "
            "baja calidad, desenfoque de movimiento brusco")


def ajustar(medida: int) -> int:
    """Redondea al múltiplo de 32 más cercano, que es lo único que acepta el modelo."""
    valido = max(MULTIPLO, round(medida / MULTIPLO) * MULTIPLO)
    if valido != medida:
        print(f"ajustada {medida} px a {valido}: el modelo solo acepta múltiplos de {MULTIPLO}")
    return valido


def clips_pedidos(args) -> list[dict]:
    """Un clip suelto (el prompt de la línea de órdenes) o los que traiga el --lote."""
    if not args.lote:
        return [{"id": Path(args.salida).stem, "prompt": args.prompt, "ruta": Path(args.salida),
                 "segundos": args.segundos, "imagen": args.imagen, "semilla": args.semilla,
                 "negativo": args.negativo}]
    trabajo = json.loads(Path(args.lote).read_text(encoding="utf-8"))
    carpeta = Path(trabajo.get("salida", "build/broll"))
    return [{"id": c["id"], "prompt": c["prompt"], "ruta": carpeta / f"{c['id']}.mp4",
             "segundos": c.get("segundos", args.segundos), "imagen": c.get("imagen", args.imagen),
             "semilla": c.get("semilla", args.semilla), "negativo": c.get("negativo", args.negativo)}
            for c in trabajo["clips"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", help="qué debe verse y cómo se mueve")
    p.add_argument("--lote", help="JSON con varios clips; carga el modelo una sola vez")
    p.add_argument("--rehacer", action="store_true", help="regenera los clips que ya existan")
    p.add_argument("--salida", default="build/broll/clip.mp4")
    p.add_argument("--imagen", help="primer fotograma (si se da, anima esa imagen)")
    p.add_argument("--segundos", type=float, default=3.0)
    p.add_argument("--alto", type=int, default=ALTO)
    p.add_argument("--ancho", type=int, default=ANCHO)
    p.add_argument("--pasos", type=int, default=50)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--negativo", default=NEGATIVO)
    p.add_argument("--modelo", default=MODELO)
    args = p.parse_args()
    if not args.prompt and not args.lote:
        p.error("hace falta un prompt, o un --lote con varios")

    alto, ancho = ajustar(args.alto), ajustar(args.ancho)
    clips = clips_pedidos(args)
    pendientes = [c for c in clips if args.rehacer or not c["ruta"].exists()]
    for c in clips:
        if c not in pendientes:
            print(f"{c['id']}: ya existe, se salta (--rehacer para regenerarlo)")
    if not pendientes:
        return 0
    # Un modelo de imagen y otro de texto no caben a la vez, así que cada lote usa uno solo.
    con_imagen = any(c["imagen"] for c in pendientes)
    if con_imagen and not all(c["imagen"] for c in pendientes):
        p.error("un lote no puede mezclar clips con imagen y sin ella: el modelo es distinto")

    # El VAE va en float32 aunque el resto vaya en bfloat16: es lo que indica el modelo.
    vae = AutoencoderKLWan.from_pretrained(args.modelo, subfolder="vae", torch_dtype=torch.float32)
    Pipeline = WanImageToVideoPipeline if con_imagen else WanPipeline
    inicio = time.time()
    pipe = Pipeline.from_pretrained(args.modelo, vae=vae, torch_dtype=torch.bfloat16)
    # Con 24 GB hay que descargar a CPU lo que no se está usando; si no, no cabe a 720p.
    pipe.enable_model_cpu_offload()
    print(f"modelo cargado en {time.time() - inicio:.0f} s · {len(pendientes)} clip(s) por delante",
          flush=True)

    filas = []
    for c in pendientes:
        # Wan trabaja en bloques de 4 fotogramas más uno: 4k+1.
        n = max(5, int(round(c["segundos"] * FPS / 4)) * 4 + 1)
        extra = {"image": load_image(str(Path(c["imagen"]).resolve()))} if c["imagen"] else {}
        inicio = time.time()
        salida = pipe(prompt=c["prompt"], negative_prompt=c["negativo"], height=alto, width=ancho,
                      num_frames=n, num_inference_steps=args.pasos, guidance_scale=args.guidance,
                      generator=torch.Generator(device="cuda").manual_seed(c["semilla"]), **extra)
        tardado = time.time() - inicio

        c["ruta"].parent.mkdir(parents=True, exist_ok=True)
        export_to_video(salida.frames[0], str(c["ruta"]), fps=FPS)
        filas.append({"id": c["id"], "prompt": c["prompt"], "ruta": str(c["ruta"]),
                      "segundos": round(n / FPS, 2), "resolucion": f"{ancho}x{alto}",
                      "pasos": args.pasos, "semilla": c["semilla"],
                      "minutos_generacion": round(tardado / 60, 1)})
        print(f"{c['ruta']} · {ancho}x{alto} · {n} fotogramas · {n / FPS:.1f} s de vídeo "
              f"· generado en {tardado / 60:.1f} min", flush=True)

    if args.lote:
        resultado = Path(json.loads(Path(args.lote).read_text(encoding="utf-8"))
                         .get("salida", "build/broll")) / "resultado.json"
        previas = json.loads(resultado.read_text(encoding="utf-8")) if resultado.exists() else []
        nuevas = {f["id"]: f for f in previas} | {f["id"]: f for f in filas}
        resultado.write_text(json.dumps(list(nuevas.values()), ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print(f"resumen en {resultado}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
