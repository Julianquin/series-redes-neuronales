"""Verifica que el entorno está listo para el pipeline en la GPU Blackwell.

Uso: .venv/bin/python env/verificar_entorno.py
"""
import shutil
import subprocess
import sys


def check(nombre, ok, detalle=""):
    print(f"[{'OK' if ok else 'FALLO'}] {nombre} {detalle}")
    return ok


resultados = []

try:
    import torch
    cuda = torch.cuda.is_available()
    resultados.append(check("PyTorch", True, torch.__version__))
    resultados.append(check("CUDA disponible", cuda, torch.version.cuda or ""))
    if cuda:
        nombre = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        resultados.append(check("GPU", True, f"{nombre} (cc {cap[0]}.{cap[1]}, {vram:.0f} GB)"))
        a = torch.randn(4096, 4096, device="cuda")
        (a @ a).sum().item()   # falla si la build no soporta Blackwell
        resultados.append(check("Operación en GPU", True))
except Exception as e:
    resultados.append(check("PyTorch/CUDA", False, repr(e)))

for binario in ["ffmpeg", "ffprobe", "latex", "dvisvgm"]:
    resultados.append(check(binario, shutil.which(binario) is not None))

try:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    resultados.append(check("NVENC (h264_nvenc)", "h264_nvenc" in out))
except Exception as e:
    resultados.append(check("NVENC", False, repr(e)))

for modulo in ["torchaudio", "manim", "faster_whisper", "chatterbox.mtl_tts", "kokoro",
               "torch_geometric", "yaml", "PIL", "jiwer"]:
    try:
        __import__(modulo)
        resultados.append(check(f"import {modulo}", True))
    except Exception as e:
        resultados.append(check(f"import {modulo}", False, repr(e)))

sys.exit(0 if all(resultados) else 1)
