# Serie de vídeos: arquitecturas de redes neuronales

## Objetivo
Pipeline automático local (RTX 5090 Laptop 24 GB, WSL2) que produce vídeos educativos en español
a partir de episodios/<ep>/guion.yaml. Orquestador: pipeline/produce.py.
Plan completo: plan_serie_redes_neuronales.md, actualizado con las decisiones tomadas; su §20 es la
bitácora (qué se cambió y por qué) y la lista de pendientes.

## Entorno Python
- Dos venvs, y no se crean más:
  - `.venv` (Python 3.13, torch 2.11.0+cu128): pipeline, Manim, demos. No se usa el `../.venv-gpu`
    compartido porque Chatterbox fija transformers==5.2.0 y diffusers==0.29.0.
  - `.venv-qwen` (Python 3.12): modelos generativos (Qwen3-TTS, faster-whisper, diffusers 0.39 para
    Wan/FLUX). La etapa 2 se ejecuta con este intérprete (`voz.python` en config/voz.yaml).
    Al instalar aquí, fija `huggingface-hub` y `transformers`: diffusers los subiría y rompería qwen-tts.
- Versiones fijadas en env/requirements.lock.txt; instalación reproducible con env/setup_wsl.sh.
- Chatterbox fija torch==2.6.0 (sin soporte para Blackwell). env/overrides.txt fuerza la build cu128.
  Instala siempre con `uv pip install --override env/overrides.txt ...`; un `pip install` normal
  podría bajar torch a 2.6 y dejar la GPU inservible.
- Audio: lee y escribe con `soundfile` (o ffmpeg). En torchaudio 2.11, `torchaudio.load/save/info` ya no
  funcionan sin TorchCodec; de torchaudio usa solo transforms/functional.
- Notebooks en notebooks/, con el kernel de `.venv`. Los procesos pesados (TTS de calibración, demos)
  van ahí para que los ejecute el usuario.

## Reglas generales
- El guion (guion.yaml) es la única fuente de verdad. Subtítulos, capítulos y marcas se derivan de él.
- Cada etapa del pipeline es un script independiente que lee y escribe archivos. Nunca cargues dos modelos grandes
  en el mismo proceso, salvo Chatterbox + Whisper en e2 (caben juntos y hacen falta para reintentar frases).
- `--simulado` (produce.py y cada etapa) sustituye los modelos por señales falsas: úsalo para probar el pipeline
  sin GPU. Sus artefactos llevan estado "simulado" y nunca deben publicarse.
- PyTorch debe ser la build cu128. Si instalas algo, ejecuta después env/verificar_entorno.py.
- Nunca uses datos de clientes ni del trabajo: solo datasets públicos o sintéticos.
- No lances entrenamientos, inferencias pesadas ni renders -qh completos por tu cuenta: entrégalos como
  comandos o notebooks que ejecuta el usuario. Renders -ql de una escena y comprobaciones ligeras sí.

## Manim
- Todas las escenas heredan de lib/escena_narrada.EscenaNarrada y se sincronizan con hasta_marca() y completar().
- Reutiliza siempre los componentes de lib/componentes/ antes de crear animaciones nuevas.
- Colores y tipografías: solo desde lib/estilo.py. Nunca uses valores hexadecimales en las escenas.
- Máximo 7 palabras por elemento de texto en pantalla. Una idea por pantalla.
- Itera con `manim -ql`. Tras cada cambio, exporta fotogramas clave a PNG y revísalos
  (solapamientos, elementos fuera de cuadro, texto ilegible, colores fuera de paleta).
  Renderiza en -qh solo cuando la revisión visual sea correcta.
- Cada componente nuevo lleva su escena de prueba en lib/pruebas/.

## Canal corto (redes_neuronales_vscode/)
- Clases verticales de ~2 min (Reel largo y Short de YouTube). Es **canal propio**, no promoción de
  la serie larga: identidad y paleta propias (dorado jerárquico), definidas en reel_common.py.
  Nunca mezcles esa paleta con la semántica de config/estilo.yaml.
- Se dibuja con Pillow, no con Manim, y se ejecuta **desde dentro de esa carpeta**: fonts/, work/ y
  out/ son rutas relativas. Usa el `.venv` de la serie; no se crea otro:
  `PATH=../.venv/bin:$PATH ./run_reels.sh`.
- Lo único que comparte con la serie es el narrador: reel_voz.narrar() llama a
  notebooks/generar_qwen.py en .venv-qwen y aplica la `velocidad` de config/voz.yaml.
- reel_voz.TimelineNarrada: la escena dura lo que tarda su frase y el dibujo se repite debajo. Las
  funciones de dibujo no cambian; reciben el fotograma del ciclo en curso.
- El listón de calidad y la anatomía del formato están en §21 del plan (dos referencias analizadas
  fotograma a fotograma). Lo esencial: código fijo en pantalla como ancla, sin cortes duros, y cada
  imagen declara en la barra de capítulo de qué línea o de qué número sale.

## Guiones
- Las marcas se listan en el orden en que se pronuncian, y llevan `ocurrencia: 2` si la palabra se
  repite. e1 lo comprueba: si no, las animaciones llegan tarde.
- Una marca por frase: sin eso, la imagen se queda quieta 10 s mientras la voz sigue.
- Tuteo, frases de menos de 25 palabras, sin relleno.
- Estructura de 7 bloques: gancho, limitacion, intuicion, matematicas, demo, cuando_usar, puente.
- Los términos técnicos en inglés van también a config/pronunciacion.yaml.

## Qué NO hacer
- No edites los archivos de salida/ a mano.
- No cambies versiones de dependencias sin avisar.
- No metas vídeo generativo en un episodio sin decidirlo antes: está en evaluación (§20 del plan).
  Para probarlo: notebooks/04_probar_wan.ipynb (usa pipeline/wan_clip.py). Todo lo visual
  publicado en la serie larga sigue siendo Manim; el canal corto se dibuja con Pillow.
