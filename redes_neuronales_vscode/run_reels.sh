#!/usr/bin/env bash
# Reels de Instagram (tipografía C). Cada reel: entrenamiento -> fotogramas (reanudable) -> codificación con audio -> out/
# Requiere haber entrenado antes (o lo hace aquí). Ejecutar desde esta carpeta.
set -euo pipefail
mkdir -p work/hd out
reel() { python3 "$1" chunk 1e12 && python3 "$1" encode; }

# Ep.1b  La noche estrellada  (salida: out/ep1b_reel_fuente_C.mp4; A y B = tipografías alternativas)
[ -f work/run_paint.pkl ] || python3 train_paint.py 6000 4096
python3 precompute_hd2.py && python3 reel_paint.py C chunk 1e12 && python3 reel_paint.py C encode
# Ep.2b  cuadros (CNN)
[ -f work/run_cuadros_rgb.pkl ]  || python3 train_cuadros.py 800 rgb
[ -f work/run_cuadros_gray.pkl ] || python3 train_cuadros.py 2000 gray
reel reel_cuadros.py
# Ep.3b  pueblos que no existen (LSTM)
[ -f work/run_pueblos.pkl ] || { python3 train_pueblos.py 3500 && python3 select_pueblos.py; }
reel reel_pueblos.py
# Ep.5b  autoencoder con cuadros
[ -f work/run_ae_cuadros.pkl ] || python3 train_ae_cuadros.py 4000
reel reel_ae.py
# Ep.6   GAN sobre el mapa de España (entrenamiento reanudable)
until [ -f work/run_gan_es.pkl ]; do python3 train_gan_es.py 1e12; done
reel reel_gan.py
# Ep.7   GNN: ¿de qué comunidad es este pueblo?
[ -f work/run_gnn_es.pkl ] || python3 train_gnn_es.py
reel reel_gnn.py
# Ep.8b  DBN con La gran ola
[ -f work/run_dbn_ola_v2.pkl ] || python3 train_dbn_ola.py
reel reel_ola.py
