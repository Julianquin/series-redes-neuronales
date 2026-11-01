"""Genera frases con Qwen3-TTS clonando la voz de referencia.

Se ejecuta con el intérprete de .venv-qwen (ver env/setup_qwen.sh), desde el notebook de
comparación, igual que el pipeline lanza cada etapa como subproceso independiente.

    .venv-qwen/bin/python notebooks/generar_qwen.py trabajo.json

`trabajo.json`: {"referencia": ruta.wav, "texto_referencia": "...", "modelo": "Qwen/...",
                 "idioma": "Spanish", "salida": carpeta,
                 "frases": [{"id": "f01", "texto": "..."}, ...]}
Escribe <salida>/<id>.wav y, al terminar, <salida>/resultado.json con duraciones y tiempos.
"""
import json
import os
import sys
import time
from pathlib import Path

# El pipeline trabaja sin conexión (los modelos de Chatterbox y Whisper ya están descargados) y esa
# variable se hereda aquí. Qwen todavía tiene que bajar el suyo, así que se quita.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # Xet se colgó dos veces en esta máquina

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main(ruta_trabajo: str) -> int:
    t = json.loads(Path(ruta_trabajo).read_text(encoding="utf-8"))
    salida = Path(t["salida"])
    salida.mkdir(parents=True, exist_ok=True)

    inicio = time.time()
    modelo = Qwen3TTSModel.from_pretrained(t.get("modelo", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                                           device_map="cuda:0", dtype=torch.bfloat16)
    print(f"modelo cargado en {time.time() - inicio:.0f} s", flush=True)

    filas = []
    for frase in t["frases"]:
        destino = salida / f"{frase['id']}.wav"
        inicio = time.time()
        wavs, sr = modelo.generate_voice_clone(
            text=frase["texto"],
            language=t.get("idioma", "Spanish"),
            ref_audio=t["referencia"],
            ref_text=t["texto_referencia"],
        )
        sf.write(destino, wavs[0], sr)
        filas.append({"id": frase["id"], "texto": frase["texto"], "ruta": str(destino),
                      "duracion": len(wavs[0]) / sr, "sr": sr,
                      "segundos_generacion": round(time.time() - inicio, 2)})
        print(f"{frase['id']} · {filas[-1]['duracion']:.2f} s", flush=True)

    (salida / "resultado.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
