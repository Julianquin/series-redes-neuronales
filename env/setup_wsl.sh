#!/usr/bin/env bash
# Prepara el entorno del proyecto en WSL2 (Ubuntu). Ejecutar desde la raíz del repo.
set -euo pipefail

# 1. Dependencias del sistema (requiere sudo). pkg-config + cairo/pango para compilar manim.
sudo apt update
sudo apt install -y pkg-config libcairo2-dev libpango1.0-dev ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-lang-spanish dvisvgm

# 2. Venv del proyecto con Python 3.13 (Chatterbox acepta numpy>=2 a partir de 3.13).
uv venv --python 3.13 .venv

# 3. Dependencias Python con las versiones fijadas. torch cu128 forzado vía overrides.txt.
uv pip install --python .venv/bin/python --override env/overrides.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r env/requirements.lock.txt

# 4. La raíz del repo en el sys.path del venv: manim, notebooks y scripts importan lib/ y pipeline/.
.venv/bin/python -c "import site; print(site.getsitepackages()[0])" | xargs -I{} sh -c 'pwd > {}/series_redes_neuronales.pth'

# 5. Verificación
.venv/bin/python env/verificar_entorno.py
