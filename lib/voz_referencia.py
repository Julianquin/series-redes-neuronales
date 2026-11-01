"""Transcribe la voz de referencia y guarda el texto en config/voz.yaml.

Qwen3-TTS clona la voz a partir del audio de referencia Y de su transcripción. En vez de
escribirla a mano (y arriesgarse a que no coincida con lo que se grabó), se obtiene con Whisper.

    .venv/bin/python -m lib.voz_referencia
"""
import sys

import yaml

from lib import config


def main() -> int:
    voz = config.cargar("voz")
    referencia = config.RAIZ / voz["referencia"]
    if not referencia.exists():
        print(f"[FALLO] no existe {voz['referencia']}")
        return 1

    import torch  # noqa: F401  (carga las librerías CUDA que faster-whisper necesita)
    from faster_whisper import WhisperModel

    modelo = config.cargar("pipeline")["voz_qa"]["modelo_whisper"]
    print(f"Transcribiendo {voz['referencia']} con Whisper {modelo}…", flush=True)
    segmentos, _ = WhisperModel(modelo, device="cuda", compute_type="float16").transcribe(
        str(referencia), language=voz["idioma"], beam_size=5)
    texto = " ".join(s.text.strip() for s in segmentos).strip()

    ruta = config.CONFIG / "voz.yaml"
    contenido = ruta.read_text(encoding="utf-8")
    datos = yaml.safe_load(contenido)
    datos["texto_referencia"] = texto
    # Se reescribe solo esa línea para no perder los comentarios del archivo.
    lineas = [f'texto_referencia: "{texto}"' if l.startswith("texto_referencia:") else l
              for l in contenido.splitlines()]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    config.cargar.cache_clear()
    print(f"\ntexto_referencia: {texto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
