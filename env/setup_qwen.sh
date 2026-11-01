#!/usr/bin/env bash
# Entorno de los modelos generativos: Qwen3-TTS (voz) y Wan 2.2 / FLUX (imagen y vídeo).
#
# Existe porque Chatterbox fija transformers==5.2.0 y diffusers==0.29.0 en el venv principal, y
# estos modelos necesitan versiones más nuevas. Todo lo generativo comparte este entorno; no se
# crean más.
#
#   ./env/setup_qwen.sh          # crea .venv-qwen e instala (los modelos se descargan al usarlos)
set -euo pipefail

# Python 3.12: es la versión que recomienda Qwen3-TTS.
uv venv --python 3.12 .venv-qwen

# torch con soporte para Blackwell (igual que en el venv principal) y después qwen-tts.
uv pip install --python .venv-qwen/bin/python \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128"
uv pip install --python .venv-qwen/bin/python --index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match qwen-tts soundfile

# QA de voz (la etapa 2 se ejecuta con este intérprete) y modelos de imagen/vídeo.
uv pip install --python .venv-qwen/bin/python --index-url https://pypi.org/simple \
  faster-whisper jiwer pyyaml soxr pyloudnorm num2words
# huggingface-hub y transformers se fijan: diffusers los subiría y rompería qwen-tts.
uv pip install --python .venv-qwen/bin/python --index-url https://pypi.org/simple \
  "diffusers>=0.35" "huggingface-hub==0.36.2" "transformers==4.57.3" ftfy imageio imageio-ffmpeg

# Comprobación: que torch siga siendo cu128 después de instalar qwen-tts.
.venv-qwen/bin/python - <<'EOF'
import torch
a = torch.randn(1024, 1024, device="cuda"); (a @ a).sum().item()
print("torch", torch.__version__, "·", torch.cuda.get_device_name(0))
import qwen_tts; print("qwen-tts importado")
EOF
