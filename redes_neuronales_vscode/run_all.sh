#!/usr/bin/env bash
# Entrena y renderiza toda la serie. Vídeos en out/, artefactos intermedios en work/.
# Tiempos medidos en 1 núcleo de CPU (sin GPU). Ejecutar desde esta carpeta.
set -euo pipefail
mkdir -p work out
run() { echo -e "\n=== $* ==="; "$@"; }

# Base: render_mlp.py es también el "tema" de la serie y CARGA work/run.pkl AL IMPORTARSE -> Ep. 1 siempre primero
run python3 train.py                      && run python3 render_mlp.py                 # Ep.1  espirales (MLP)
run python3 train_paint.py 6000 4096      && run python3 precompute_paint.py && run python3 render_paint.py   # Ep.1b La noche estrellada (~5+3+3 min)
run python3 train_cuadros.py 2000 gray    && run python3 train_cuadros.py 800 rgb && run python3 render_cuadros.py  # Ep.2b cuadros (CNN)
run python3 train_pueblos.py 3500         && run python3 select_pueblos.py && run python3 render_pueblos.py      # Ep.3b pueblos (~5 min)
run python3 train_attn.py                 && run python3 render_attn.py                # Ep.4  atención
run python3 train_ae_cuadros.py 4000      && run python3 render_ae_cuadros.py          # Ep.5b autoencoder cuadros
run python3 train_gan.py                  && run python3 render_gan.py                 # Ep.6  GAN
run python3 train_gnn.py                  && run python3 render_gnn.py                 # Ep.7  GNN (club de kárate)
run python3 train_dbn_ola.py              && run python3 render_dbn_ola.py             # Ep.8b DBN La gran ola (~4 min)

# Versiones clásicas (sustituidas por las "b"); descomentar si se quieren regenerar
# run python3 train_cnn.py  && run python3 render_cnn.py    # Ep.2 dígitos
# run python3 train_lstm.py && run python3 render_lstm.py   # Ep.3 LSTM texto
# run python3 train_ae.py   && run python3 render_ae.py     # Ep.5 dígitos
# run python3 train_dbn.py  && run python3 render_dbn.py    # Ep.8 dígitos
