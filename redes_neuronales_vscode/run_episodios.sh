#!/usr/bin/env bash
# Renderiza los 8 episodios largos (explainer/). Requiere los entrenamientos de run_reels.sh / run_all.sh.
# Si existe explainer/voz/timings.json (tts.py), usa los tiempos reales de la voz.
set -euo pipefail
mkdir -p work out
for ep in ep01_mlp ep02_cnn ep03_rnn ep04_transformer ep05_ae ep06_gan ep07_gnn ep08_dbn; do
  echo "=== $ep ==="
  python3 explainer/$ep.py chunk 1e12
  python3 explainer/$ep.py encode
  rm -rf work/frames_${ep}_piloto          # ~0,6–1 GB por episodio
done
