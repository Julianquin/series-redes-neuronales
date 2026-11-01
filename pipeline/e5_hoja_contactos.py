"""Etapa 5 — Hoja de contactos: fotogramas clave de cada segmento en una imagen (§10.7).

Por segmento: inicio, mitad y final, más un fotograma justo después de cada marca (donde más
fallan las escenas). Cada fila lleva el id, la escena y el tiempo. Es el punto de control 2:
en 2–3 minutos se ven textos solapados, elementos fuera de cuadro o colores raros.

Salida: build/contactos/<calidad>/hoja_<n>.png (varias páginas si el episodio es largo).

    python -m pipeline.e5_hoja_contactos --ep 99 [--borrador]
"""
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lib import config
from pipeline import comun

ANCHO_MINI = 480
FILAS_POR_PAGINA = 8
MARGEN = 12
ALTO_ETIQUETA = 30
FONDO = (40, 42, 48)
TEXTO = (232, 234, 237)
SECUNDARIO = (154, 160, 166)


def fuente(tam: int) -> ImageFont.FreeTypeFont:
    for ruta in ("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(ruta).exists():
            return ImageFont.truetype(ruta, tam)
    return ImageFont.load_default()


def instantes(duracion: float, marcas: dict[str, float]) -> list[tuple[str, float]]:
    """(etiqueta, segundo) de los fotogramas a extraer, ordenados y sin duplicados cercanos."""
    puntos = [("inicio", min(0.5, duracion * 0.1)), ("mitad", duracion / 2), ("final", max(0.0, duracion - 0.1))]
    # 0,4 s después de la marca: la animación ya ha arrancado.
    puntos += [(evento, min(t + 0.4, duracion - 0.1)) for evento, t in marcas.items()]
    puntos.sort(key=lambda p: p[1])
    limpios = []
    for etiqueta, t in puntos:
        if limpios and abs(t - limpios[-1][1]) < 0.3:
            if etiqueta not in ("inicio", "mitad", "final"):   # la marca es más informativa
                limpios[-1] = (etiqueta, t)
            continue
        limpios.append((etiqueta, t))
    return limpios


def extraer(video: Path, t: float, destino: Path) -> Image.Image:
    comun.ffmpeg("-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={ANCHO_MINI}:-2", str(destino))
    return Image.open(destino).convert("RGB")


def componer(filas: list[dict], destino: Path) -> None:
    columnas = max(len(f["fotos"]) for f in filas)
    alto_mini = filas[0]["fotos"][0][1].height
    ancho = MARGEN + columnas * (ANCHO_MINI + MARGEN)
    alto_fila = ALTO_ETIQUETA + alto_mini + 24 + MARGEN
    hoja = Image.new("RGB", (ancho, MARGEN + len(filas) * alto_fila), FONDO)
    d = ImageDraw.Draw(hoja)
    f_titulo, f_pie = fuente(18), fuente(14)
    y = MARGEN
    for fila in filas:
        d.text((MARGEN, y + 4), f"{fila['id']} · {fila['escena']} · {fila['duracion']:.1f} s",
               font=f_titulo, fill=TEXTO)
        for c, (etiqueta, img, t) in enumerate(fila["fotos"]):
            x = MARGEN + c * (ANCHO_MINI + MARGEN)
            hoja.paste(img, (x, y + ALTO_ETIQUETA))
            d.text((x, y + ALTO_ETIQUETA + alto_mini + 4), f"{t:.2f} s · {etiqueta}", font=f_pie, fill=SECUNDARIO)
        y += alto_fila
    hoja.save(destino)


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    calidad = comun.calidad(args)
    dir_render = dir_ep / "build" / "render" / calidad
    dir_salida = comun.dir_build(dir_ep, "contactos", calidad)
    for vieja in dir_salida.glob("hoja_*.png"):
        vieja.unlink()

    segmentos = [s for s in guion["segmentos"] if not args.segmento or s["id"] == args.segmento]
    faltan = [s["id"] for s in segmentos if not (dir_render / f"{s['id']}.mp4").exists()]
    if faltan:
        print(f"[FALLO] faltan vídeos de {', '.join(faltan)} en {dir_render.relative_to(config.RAIZ)}: "
              f"ejecuta antes e4{' --borrador' if args.borrador else ''}")
        return comun.FALLO

    filas = []
    with tempfile.TemporaryDirectory() as tmp:
        for seg in segmentos:
            video = dir_render / f"{seg['id']}.mp4"
            meta = json.loads((dir_ep / "build" / "audio" / f"{seg['id']}.json").read_text(encoding="utf-8"))
            duracion = comun.duracion_video(video)
            fotos = [(etiqueta, extraer(video, t, Path(tmp) / f"{seg['id']}_{i}.png"), t)
                     for i, (etiqueta, t) in enumerate(instantes(duracion, meta["marcas"]))]
            filas.append({"id": seg["id"], "escena": seg["escena"], "duracion": duracion, "fotos": fotos})

    paginas = [filas[i:i + FILAS_POR_PAGINA] for i in range(0, len(filas), FILAS_POR_PAGINA)]
    for n, pagina in enumerate(paginas, 1):
        destino = dir_salida / f"hoja_{n}.png"
        componer(pagina, destino)
        print(f"[OK] {destino.relative_to(config.RAIZ)} ({', '.join(f['id'] for f in pagina)})")
    print("Control 2: revisa la hoja (solapamientos, elementos fuera de cuadro, colores) antes de ensamblar.")
    return comun.OK


if __name__ == "__main__":
    sys.exit(main())
