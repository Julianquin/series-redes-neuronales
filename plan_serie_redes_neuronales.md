# Plan de producción: serie de vídeos sobre arquitecturas de redes neuronales

**Pipeline automático, 100 % local y sin presupuesto, sobre una NVIDIA RTX 5090 Laptop (24 GB)**

> Actualizado el 2026-09-20 con las decisiones tomadas durante la construcción. Los cambios
> respecto a la versión inicial están marcados con «(actualizado)» y resumidos en §20.

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Objetivos y principios](#2-objetivos-y-principios)
3. [Restricciones y hardware](#3-restricciones-y-hardware)
4. [Temario de la serie](#4-temario-de-la-serie)
5. [Estructura estándar de cada vídeo](#5-estructura-estándar-de-cada-vídeo)
6. [Guía de estilo visual y narrativo](#6-guía-de-estilo-visual-y-narrativo)
7. [Stack tecnológico y licencias](#7-stack-tecnológico-y-licencias)
8. [Arquitectura del pipeline](#8-arquitectura-del-pipeline)
9. [Formato del guion (`guion.yaml`)](#9-formato-del-guion-guionyaml)
10. [Etapas del pipeline en detalle](#10-etapas-del-pipeline-en-detalle)
11. [Librería de componentes Manim](#11-librería-de-componentes-manim)
12. [Puntos de control humanos](#12-puntos-de-control-humanos)
13. [Configuración del entorno](#13-configuración-del-entorno)
14. [Trabajo con Claude Code](#14-trabajo-con-claude-code)
15. [Hoja de ruta](#15-hoja-de-ruta)
16. [Riesgos y mitigaciones](#16-riesgos-y-mitigaciones)
17. [Políticas de YouTube, ética y transparencia](#17-políticas-de-youtube-ética-y-transparencia)
18. [Checklist por episodio](#18-checklist-por-episodio)
19. [Anexos](#19-anexos)
20. [Bitácora de decisiones y pendientes](#20-bitácora-de-decisiones-y-pendientes)
21. [Referencias de calidad y formato de clase corta](#21-referencias-de-calidad-y-formato-de-clase-corta)

---

## 1. Resumen ejecutivo

El objetivo es producir una serie de vídeos educativos en español, de alta calidad, que expliquen las principales arquitecturas de redes neuronales: MLP, CNN, RNN/LSTM/GRU, Transformers, Autoencoders, GAN, Diffusion, GNN y, como capítulo histórico, DBN.

La producción se apoya en un **pipeline automatizado** que se ejecuta con un solo comando:

```
guion.yaml → voz sintética (TTS) → escenas Manim sincronizadas → render
          → ensamblado (ffmpeg) → subtítulos → control de calidad → vídeo final
```

Decisiones clave:

- **Animación:** Manim Community con una librería propia de componentes reutilizables. Es lo que determina la calidad educativa.
- **Voz:** Chatterbox Multilingual (licencia MIT, compatible con monetizar), corriendo en local con una voz de referencia fija.
- **Sincronización:** el audio se genera primero, frase a frase. Las animaciones leen las duraciones y las "marcas" (momentos exactos en que se pronuncia una palabra) desde un JSON.
- **Montaje:** ffmpeg, en lugar de DaVinci Resolve, porque la API de scripting de DaVinci solo existe en la versión de pago.
- **Supervisión humana mínima:** 3 puntos de control de unos minutos cada uno (guion, hoja de fotogramas y vídeo final).
- **Construcción:** Claude Code construye el pipeline una vez y después genera las escenas de cada episodio.

Tiempo estimado por episodio, una vez el sistema esté maduro: **2–4 horas de trabajo humano**, más varias horas de render desatendido.

---

## 2. Objetivos y principios

### Objetivos

1. Explicar cada arquitectura desde la **intuición visual** antes que desde la fórmula.
2. Mostrar que cada arquitectura **resuelve una limitación de la anterior**. Ese es el hilo conductor de la serie.
3. Incluir en cada vídeo una **demo real**, entrenada en la RTX 5090, con resultados honestos.
4. Mantener una **identidad visual y sonora constante** en toda la serie.
5. Reducir el coste marginal de cada episodio mediante automatización y reutilización.

### Principios

- **Una idea por pantalla.** Si una escena necesita dos ideas, se divide en dos.
- **Las matemáticas llegan después de la intuición**, y solo las imprescindibles.
- **Honestidad técnica.** Se dice cuándo una arquitectura *no* es la mejor opción (por ejemplo, un MLP frente a gradient boosting en datos tabulares).
- **Automatizar lo repetitivo, supervisar lo que afecta a la credibilidad.** Un error conceptual o una pronunciación absurda cuestan más que 10 minutos de revisión.
- **Nada de datos de clientes ni del trabajo.** Todas las demos usan datasets públicos o datos sintéticos.

---

## 3. Restricciones y hardware

| Recurso | Situación | Implicación |
|---|---|---|
| Presupuesto | 0 € | Solo software gratuito y open source, y modelos con licencia compatible con monetizar |
| GPU | RTX 5090 Laptop, 24 GB VRAM, arquitectura Blackwell (actualizado) | Requiere **PyTorch compilado para CUDA 12.8 o superior** (builds `cu128`). Las versiones antiguas fallan con *"no kernel image is available"* |
| VRAM | 24 GB | Los modelos no se cargan a la vez: cada etapa se ejecuta como **subproceso independiente** que libera la memoria al terminar |
| CPU | Determinante para Manim | El renderizador Cairo de Manim usa sobre todo la **CPU**, no la GPU. Se paraleliza por segmentos |
| Tiempo humano | Limitado | El sistema gira en torno a los 3 puntos de control |

**Qué aporta la GPU en este proyecto:** TTS, alineado con Whisper, demos de entrenamiento, generación de miniaturas y codificación de vídeo con NVENC.

**Qué no aporta:** el render de Manim, que es mayoritariamente CPU.

---

## 4. Temario de la serie

### 4.1 Visión general

| Ep. | Tema | Pregunta que resuelve | Duración objetivo |
|---|---|---|---|
| 00 | MLP / Feedforward | ¿Cómo aprende una red una función a partir de datos? | 12–14 min |
| 01 | CNN | ¿Por qué un MLP falla con imágenes? | 12–15 min |
| 02 | RNN, LSTM, GRU | ¿Cómo recordar el pasado en una secuencia? | 12–15 min |
| 03 | Transformers | ¿Cómo mirar toda la secuencia a la vez? | 15–18 min |
| 04 | Autoencoders | ¿Se puede aprender sin etiquetas comprimiendo? | 10–12 min |
| 05 | VAE → GAN | ¿Cómo generar datos nuevos? | 12–14 min |
| 06 | Diffusion | ¿Por qué los modelos de difusión desplazaron a las GAN? | 12–15 min |
| 07 | GNN | ¿Qué hacer cuando los datos son un grafo? | 12–14 min |
| Extra | DBN / RBM | ¿Cómo empezó el deep learning moderno (2006)? | 6–8 min |

El capítulo extra se puede publicar en cualquier momento. Encaja bien entre el 00 y el 01, como pausa histórica.

### 4.2 Detalle por episodio

#### Episodio 00 — MLP: qué aprende una red neuronal

- **Idea central:** una red neuronal es una composición de funciones sencillas que **deforma el espacio** hasta que los datos se vuelven separables.
- **Secuencia visual:**
  1. Nube de puntos en dos clases (forma de lunas). Una recta intenta separarlos y falla.
  2. Una neurona: suma ponderada, sesgo y activación (ReLU), construida paso a paso.
  3. Una capa como transformación del plano: la cuadrícula se estira, se dobla y se pliega.
  4. La función de pérdida como paisaje 3D, con una bola que desciende por el gradiente.
  5. Backpropagation como gradientes que fluyen hacia atrás por la red (regla de la cadena).
  6. La frontera de decisión deformándose durante un **entrenamiento real**, a partir de los logs de la demo.
- **Matemáticas mínimas:** `y = σ(Wx + b)`, la función de pérdida, la regla de la cadena y la actualización `w ← w − η·∂L/∂w`.
- **Demo:** MLP en PyTorch sobre `make_moons`. Después, una comparación honesta en un dataset tabular público: MLP frente a gradient boosting. Lo habitual es que gane gradient boosting, y hay que decirlo.
- **Errores a evitar:** decir que "imita al cerebro", ignorar la normalización de las entradas y presentar el MLP como la mejor opción en tabular.
- **Puente al siguiente episodio:** "Ahora dale a este MLP una foto de 224×224 píxeles…"

#### Episodio 01 — CNN: ver con filtros

- **Idea central:** en una imagen, lo importante es **local y se repite**. La convolución aprovecha ambas cosas compartiendo pesos.
- **Secuencia visual:**
  1. Una imagen de 224×224×3 aplanada en 150 528 entradas: el MLP pierde la estructura espacial y el número de pesos explota.
  2. Un kernel 3×3 deslizándose sobre la imagen y produciendo un mapa de características.
  3. Pesos compartidos: el mismo detector de bordes en todas las posiciones.
  4. Stride y pooling reduciendo la resolución.
  5. Campo receptivo creciendo con la profundidad; jerarquía de bordes → texturas → partes → objetos.
  6. Más allá de clasificar: detección (cajas), segmentación (píxeles) y audio (espectrogramas como imágenes).
- **Matemáticas mínimas:** la convolución discreta como suma de productos y el cálculo del tamaño de salida.
- **Demo:** ResNet preentrenada. Visualización de activaciones por capa y un mapa Grad-CAM de "dónde mira la red".
- **Puente al siguiente episodio:** "Las imágenes tienen estructura en el espacio. ¿Y los datos que tienen estructura en el tiempo?"

#### Episodio 02 — RNN, LSTM y GRU: memoria en secuencias

- **Idea central:** un estado oculto que se actualiza paso a paso funciona como memoria, pero esa memoria se degrada.
- **Secuencia visual:**
  1. Una secuencia (palabras o una serie temporal) entrando elemento a elemento.
  2. La RNN como bucle y la misma red "desenrollada" en el tiempo.
  3. Backpropagation through time: gradientes que se desvanecen al retroceder muchos pasos.
  4. La LSTM: la celda como una cinta transportadora y las puertas de olvido, entrada y salida abriéndose y cerrándose.
  5. La GRU como versión simplificada de la LSTM.
- **Matemáticas mínimas:** `h_t = tanh(W·h_{t−1} + U·x_t + b)` y la intuición de multiplicar muchos números menores que 1.
- **Demo:** predicción de una serie temporal pública o sintética con estacionalidad. Una LSTM frente a un *baseline* ingenuo estacional, con comparación honesta de métricas.
- **Puente al siguiente episodio:** "¿Y si, en lugar de recordar paso a paso, pudiéramos mirar toda la secuencia a la vez?"

#### Episodio 03 — Transformers: atención

- **Idea central:** cada elemento de la secuencia decide **a qué otros elementos prestar atención**, y todos se procesan en paralelo.
- **Secuencia visual:**
  1. Las limitaciones de la RNN: procesa en secuencia y su memoria se degrada.
  2. La analogía de la búsqueda: *query* (lo que busco), *key* (lo que ofrezco) y *value* (lo que entrego).
  3. La matriz `QKᵀ` construyéndose celda a celda, seguida del escalado, el softmax y la multiplicación por `V`.
  4. Multi-head: varias "miradas" en paralelo sobre la misma frase.
  5. Codificación posicional: por qué hace falta.
  6. El bloque Transformer completo (atención, feed-forward, residuales, normalización).
  7. Generación token a token con probabilidades, y ViT: una imagen como secuencia de parches.
- **Matemáticas mínimas:** `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`.
- **Demo:** un modelo pequeño multilingüe mostrando mapas de atención reales sobre una frase en español. Después, un LLM pequeño en local mostrando la distribución de probabilidad del siguiente token.
- **Puente al siguiente episodio:** "Hasta ahora siempre hemos tenido etiquetas. ¿Qué se puede aprender sin ellas?"

#### Episodio 04 — Autoencoders: comprimir para entender

- **Idea central:** si obligas a una red a reconstruir su entrada pasando por un cuello de botella, aprende lo esencial de los datos.
- **Secuencia visual:**
  1. Encoder → cuello de botella → decoder.
  2. Espacio latente 2D con los dígitos de MNIST agrupándose por clase sin haber visto etiquetas.
  3. La relación con PCA: un autoencoder lineal equivale aproximadamente a PCA.
  4. Denoising autoencoder: reconstruir la entrada limpia a partir de una ruidosa.
  5. Detección de anomalías: los datos "raros" se reconstruyen mal.
- **Demo:** autoencoder entrenado sin una de las clases. El error de reconstrucción de esa clase se dispara, y ahí está el detector de anomalías.
- **Puente al siguiente episodio:** "Si el decoder sabe reconstruir… ¿puede inventar?"

#### Episodio 05 — Del VAE a las GAN: generar datos nuevos

- **Idea central:** para generar hace falta un espacio latente sin huecos (VAE) o un "crítico" que empuje al generador hacia el realismo (GAN).
- **Secuencia visual:**
  1. Muestrear en el espacio latente de un autoencoder produce basura, porque tiene huecos.
  2. El VAE: cada punto se convierte en una distribución y el espacio latente queda continuo.
  3. La GAN como juego entre falsificador y detective.
  4. Evolución del entrenamiento: del ruido a dígitos reconocibles.
  5. Los problemas: inestabilidad y *mode collapse* (el generador aprende a producir siempre lo mismo).
  6. Aplicaciones: rostros sintéticos, aumento de datos y transferencia de estilo.
- **Matemáticas mínimas:** el juego minimax, explicado en palabras y con una fórmula.
- **Demo:** DCGAN sobre Fashion-MNIST, con una rejilla de muestras por época convertida en animación.
- **Puente al siguiente episodio:** "¿Y si, en vez de enfrentar dos redes, enseñamos a una sola a quitar ruido?"

#### Episodio 06 — Modelos de difusión

- **Idea central:** destruir una imagen con ruido es fácil. Si una red aprende a **deshacer cada paso**, puede crear imágenes partiendo del ruido puro.
- **Secuencia visual:**
  1. Proceso directo: una imagen se va llenando de ruido hasta ser ruido puro.
  2. La red aprende a predecir el ruido que se añadió.
  3. Proceso inverso: del ruido a la imagen, paso a paso.
  4. Guía por texto: la intuición del condicionamiento y de la *classifier-free guidance*.
  5. Latent diffusion: difundir en el espacio latente de un autoencoder, lo que conecta con el episodio 04.
  6. Por qué desplazaron a las GAN: estabilidad y diversidad, a cambio de más pasos de cómputo.
- **Demo:** FLUX.1 [schnell] o Stable Diffusion en local, mostrando las imágenes intermedias de cada paso.
- **Puente al siguiente episodio:** "Imágenes, texto, secuencias… ¿y si los datos son una red de relaciones?"

#### Episodio 07 — Graph Neural Networks

- **Idea central:** cada nodo aprende **escuchando a sus vecinos**. Cada capa amplía el radio de escucha en un salto.
- **Secuencia visual:**
  1. Datos que son grafos por naturaleza: redes sociales, moléculas, compras entre usuarios y productos.
  2. Por qué no sirve una tabla: el orden de los nodos es arbitrario y cada nodo tiene un número distinto de vecinos.
  3. *Message passing*: los vecinos envían mensajes, el nodo los agrega y se actualiza.
  4. Varias capas equivalen a varios saltos de vecindad.
  5. GCN frente a GAT (atención sobre vecinos, lo que conecta con el episodio 03).
  6. Aplicaciones: recomendación, descubrimiento de fármacos y detección de fraude.
- **Demo:** GCN sobre el dataset Cora con PyTorch Geometric, y los *embeddings* proyectados con t-SNE antes y después de entrenar.
- **Cierre de la serie:** un mapa final con todas las arquitecturas conectadas por las limitaciones que resuelven.

#### Extra — DBN y RBM: cómo empezó todo

- **Idea central:** en 2006 no se sabía entrenar redes profundas. Las DBN lo resolvieron pre-entrenando capa a capa.
- **Secuencia visual:** una línea temporal (perceptrón → invierno de la IA → 2006 → 2012 con AlexNet), la RBM como modelo de energía (en intuición), el pre-entrenamiento *greedy* capa a capa, y por qué quedó obsoleto: ReLU, buena inicialización, GPUs, más datos y BatchNorm.
- **Demo:** ninguna. Es un capítulo narrativo.

---

## 5. Estructura estándar de cada vídeo

| Bloque | Duración | Contenido | Tipo de escena |
|---|---|---|---|
| 1. Gancho | 0:00–0:30 | Problema concreto o resultado sorprendente, sin definiciones | Manim, impacto visual |
| 2. Limitación anterior | ~1 min | Por qué no basta la arquitectura del episodio previo | Manim, reutilizando componentes |
| 3. Intuición visual | 4–5 min | El núcleo del episodio, sin fórmulas | Manim, lo más cuidado |
| 4. Matemáticas mínimas | 2–3 min | Solo las ecuaciones esenciales, animadas término a término | Manim + LaTeX |
| 5. Demo real | 2–3 min | Código, entrenamiento y resultados | Escena de demo automatizada (ver §10.5) |
| 6. Cuándo usarla y cuándo no | ~1 min | Casos reales y alternativas | Manim, tarjetas comparativas |
| 7. Puente | ~15 s | Pregunta que abre el siguiente episodio | Manim |

Los bloques se convierten automáticamente en **capítulos de YouTube** (ver §10.8).

---

## 6. Guía de estilo visual y narrativo

### 6.1 Visual

- **Resolución y cadencia:** 1920×1080 a 60 fps (`manim -qh`). El 4K (`-qk`) es opcional y multiplica el tiempo de render.
- **Fondo:** gris muy oscuro (por ejemplo, `#111318`), no negro puro. Cansa menos y evita falsos positivos en la detección de pantallas negras.
- **Paleta semántica**, igual en toda la serie:

| Concepto | Color |
|---|---|
| Datos y entradas | Azul |
| Activaciones y flujo hacia delante | Verde azulado |
| Gradientes y flujo hacia atrás | Naranja |
| Pesos y parámetros | Violeta |
| Error y pérdida | Rojo |
| Atención | Amarillo |

- **Tipografía:** una sans-serif libre (Inter o Source Sans 3) para texto y LaTeX para ecuaciones.
- **Texto en pantalla:** un máximo de 7 palabras por elemento. La narración explica y la pantalla muestra.
- **Movimiento:** transiciones suaves y sin efectos gratuitos. Cada animación responde a una frase de la narración.

### 6.2 Narrativa

- Tuteo, tono cercano pero preciso.
- Frases cortas, porque el TTS rinde mejor con frases de menos de 25 palabras.
- Los términos en inglés se introducen una vez con su explicación ("el *gradiente*, es decir…") y después se usan con naturalidad.
- Nada de relleno: si una frase no aporta, se elimina.

Toda la configuración de estilo vive en `config/estilo.yaml` y la importa `lib/estilo.py`. Así, ninguna escena define colores "a mano".

El **formato de clase corta** (vertical, dos minutos) tiene su propia identidad y su propia lógica
de color, deliberadamente distinta de esta: ver §21.

---

## 7. Stack tecnológico y licencias

El criterio de selección es que todo sea gratuito, corra en local y tenga una **licencia compatible con monetizar** el canal.

| Función | Herramienta | Licencia | Notas |
|---|---|---|---|
| Animación | Manim Community | MIT | Núcleo visual del proyecto |
| Voz (TTS) | **Qwen3-TTS** (Alibaba), modelo `12Hz-1.7B-Base` | Apache 2.0 | **Motor actual (actualizado 2026-09-20).** Clona la voz a partir del audio de referencia **y su transcripción**. Notablemente más natural en español que Chatterbox en nuestra prueba |
| Voz (alternativa) | Chatterbox Multilingual (Resemble AI) | MIT | Motor anterior; sigue disponible en `config/voz.yaml`. Añadía "oh" y colas de sonido, y acento inglés en palabras como "red" |
| Voz de referencia | Kokoro-82M (voces en español) | Apache 2.0 | Solo para crear el audio de referencia, que después se clona con Chatterbox (ver §10.3) |
| Alineado y QA de audio | faster-whisper (large-v3) | MIT | Marcas de tiempo por palabra y detección de errores de pronunciación |
| Montaje y codificación | ffmpeg + NVENC | LGPL/GPL | Concatenación, mezcla de música y exportación |
| Guion y generación de escenas | Claude Code | Suscripción existente | Construye el pipeline y escribe las escenas |
| LLM local (opcional) | Qwen3-14B o gpt-oss-20b vía Ollama | Apache 2.0 | Tareas auxiliares: metadatos, revisión de estilo |
| Miniaturas | **FLUX.2 [klein] 4B** (o FLUX.1 [schnell]) | Apache 2.0 | Solo la variante de 4B es Apache 2.0; la de 9B es no comercial. Pendiente de implementar (etapa 9) |
| Vídeo generativo (en evaluación) | Wan 2.2 TI2V-5B / I2V-A14B | Apache 2.0 | **Cambio respecto a la versión inicial**, que lo descartaba. Se prueba como material de apoyo puntual; decisión tras ver la retención del piloto (§20) |
| Demos | PyTorch (cu128), PyTorch Geometric, Hugging Face | BSD / MIT / Apache | |
| Música | Biblioteca de audio de YouTube | Según la pista | Algunas pistas exigen atribución en la descripción: guardar la atribución junto al archivo |
| Retoque manual (opcional) | Krita, DaVinci Resolve (gratis) | Gratuitos | Solo si se quiere retocar algo a mano |

### Motores descartados y por qué

- **XTTS-v2:** buena calidad en español, pero su licencia (Coqui Public Model License) **no permite uso comercial**. Si el canal se monetiza, queda descartado.
- **Modelos base de F5-TTS:** licencia CC-BY-NC, también no comercial.
- **Kokoro como voz principal:** su español es claramente inferior a su inglés. Solo se usa para generar la voz de referencia.
- **Vídeo generativo (Wan, LTX-Video):** *descartado en la versión inicial* por aportar poco en
  divulgación técnica, consumir mucha GPU y verse "IA". **En evaluación desde el 2026-09-20**: se
  prueba Wan 2.2 TI2V-5B para planos de apoyo, con reglas estrictas (sin texto generado, sin
  personas realistas) y decisión basada en la retención del piloto.

> Las licencias cambian. Antes de monetizar, revisa la licencia vigente de cada modelo en su repositorio oficial y guarda una copia en `docs/licencias/`.

---

## 8. Arquitectura del pipeline

### 8.1 Flujo

```
                   ┌───────────────────────────┐
                   │ guion.yaml (episodio)     │ ◄── Punto de control 1
                   └─────────────┬─────────────┘
                                 ▼
  Etapa 1  Validar guion ──► esquema, longitudes, componentes existentes
                                 ▼
  Etapa 2  TTS por frase ──► audio + duraciones + marcas (JSON)
                                 │   QA de voz: Whisper (WER), duración/carácter
                                 ▼
  Etapa 3  Escenas Manim ──► escenas.py (Claude Code) usando lib/ + JSON de audio
                                 ▼
  Etapa 4  Render ──► un .mp4 por segmento (en paralelo, CPU)
                                 ▼
  Etapa 5  Hoja de contactos ──► fotogramas clave en una imagen ◄── Punto de control 2
                                 ▼
  Etapa 6  Ensamblado ──► concat + música con ducking + loudnorm + NVENC
                                 ▼
  Etapa 7  Subtítulos y capítulos ──► .srt desde el guion + capítulos desde los bloques
                                 ▼
  Etapa 8  QA final ──► negros, congelados, silencios, sonoridad, duraciones
                                 ▼
  Etapa 9  Miniatura y metadatos ──► ComfyUI + título/descripción ◄── Punto de control 3
                                 ▼
                         salida/epXX/  (listo para subir)
```

### 8.2 Principios de diseño

- **Cada etapa es un subproceso independiente** que lee archivos y escribe archivos. Así la VRAM se libera entre etapas y cualquier etapa se puede relanzar sola.
- **Idempotencia y caché:** cada artefacto guarda el *hash* de sus entradas. Si el texto de una frase no cambia, no se vuelve a generar su audio; si una escena no cambia, no se vuelve a renderizar.
- **El guion es la única fuente de verdad.** Los subtítulos, los capítulos y las marcas se derivan de `guion.yaml`, nunca se escriben a mano.
- **Fallar pronto y con claridad:** si algo no pasa el QA, el pipeline se detiene e indica qué segmento revisar.

### 8.3 Estructura del repositorio

```
series-redes-neuronales/
├── CLAUDE.md                     # contexto para Claude Code (ver Anexo A)
├── README.md
├── pyproject.toml
├── config/
│   ├── estilo.yaml               # paleta, tipografías, tamaños
│   ├── voz.yaml                  # motor TTS, referencia, semilla, parámetros
│   ├── pronunciacion.yaml        # diccionario de términos técnicos
│   └── pipeline.yaml             # umbrales de QA, rutas, calidad de render
├── env/
│   ├── setup_wsl.sh              # sistema + .venv (pipeline)
│   ├── setup_qwen.sh             # .venv-qwen (voz, imagen y vídeo generativos)
│   ├── overrides.txt             # fuerza torch cu128 frente a los pines de los modelos
│   ├── requirements.lock.txt
│   └── verificar_entorno.py      # ver Anexo B
├── notebooks/                    # calibrar voz, probar pronunciaciones, comparar motores
├── lib/
│   ├── config.py                 # rutas del repo y carga de YAML
│   ├── estilo.py
│   ├── texto.py                  # frases, pausas, pronunciación, cifras a palabras
│   ├── audio.py                  # recortes, sonoridad, remuestreo (soundfile, no torchaudio)
│   ├── musica.py                 # tempo, subidas de energía y secciones de cada pista
│   ├── ritmo.py                  # modelo de duración del narrador, por motor
│   ├── voz_referencia.py         # transcribe narrador.wav (lo necesita Qwen3-TTS)
│   ├── escena_narrada.py         # EscenaNarrada y EscenaNarrada3D (ver §10.4)
│   ├── pruebas/                  # una escena de prueba por componente
│   └── componentes/
│       ├── neurona.py
│       ├── capa.py
│       ├── red_densa.py
│       ├── flecha_gradiente.py
│       ├── matriz.py
│       ├── kernel_convolucion.py
│       ├── mapa_atencion.py
│       ├── grafo.py
│       ├── paisaje_perdida.py
│       ├── bloque_codigo.py
│       └── grafica_entrenamiento.py
├── pipeline/
│   ├── produce.py                # orquestador: python produce.py --ep 00
│   ├── e1_validar_guion.py
│   ├── e2_tts.py
│   ├── e4_render.py
│   ├── e5_hoja_contactos.py
│   ├── e6_ensamblar.py
│   ├── e7_subtitulos_capitulos.py
│   ├── e8_qa.py
│   ├── e9_miniatura_metadatos.py
│   └── wan_clip.py               # herramienta suelta: vídeo generativo en evaluación (§20)
├── episodios/
│   └── ep00_mlp/
│       ├── guion.yaml
│       ├── escenas.py
│       ├── demo/                 # entrenamiento real + logs
│       └── build/                # artefactos intermedios (ignorado en git)
├── assets/
│   ├── musica/                   # pistas + atribuciones
│   ├── voz_referencia/
│   └── fuentes/
├── docs/
│   └── licencias/
└── salida/
    └── ep00/                     # vídeo final, .srt, miniatura, metadatos.md
```

---

## 9. Formato del guion (`guion.yaml`)

El guion es **datos estructurados**, no texto libre. Cada segmento corresponde a una escena de Manim.

```yaml
episodio: 0
slug: ep00_mlp
titulo: "Qué aprende realmente una red neuronal"
duracion_objetivo_min: 13
musica: assets/musica/fondo_01.mp3

segmentos:
  - id: s01
    bloque: gancho                     # gancho | limitacion | intuicion | matematicas | demo | cuando_usar | puente
    escena: GanchoFrontera             # nombre de la clase en escenas.py
    narracion: >
      Mira estos puntos. Ninguna línea recta puede separarlos.
      [pausa:0.4] Y sin embargo, esta red va a encontrar la frontera ella sola.
    visual: >
      Nube de puntos en forma de dos lunas, azul y naranja. Aparece una recta que falla.
      En la marca "frontera" la recta empieza a curvarse hasta separar ambas clases.
    componentes: [NubePuntos, FronteraDecision]
    marcas:
      - palabra: "frontera"
        evento: frontera_empieza

  - id: s02
    bloque: intuicion
    escena: NeuronaPasoAPaso
    narracion: >
      Empecemos por la pieza más pequeña: una neurona.
      Recibe unos números, multiplica cada uno por un peso, los suma
      y pasa el resultado por una función de activación.
    visual: >
      Neurona central. Entran tres flechas con valores; los pesos aparecen en violeta
      al decir "peso"; el sumatorio al decir "suma"; la curva ReLU al decir "activación".
    componentes: [Neurona, FlechaEntrada]
    marcas:
      - {palabra: "peso", evento: mostrar_pesos}
      - {palabra: "suma", evento: mostrar_suma}
      - {palabra: "activación", evento: mostrar_relu}
```

**Música por tramos (actualizado).** El campo `musica` admite una lista, para que cada pista entre
en el bloque que le toca y su transición caiga en un cambio de segmento:

```yaml
musica:
  - pista: "assets/musica/intergalactic.mp3"
    desde: s01
    entrada: s02        # la subida de la pista cae al empezar s02
    transicion: auto    # 'auto' = primera subida de energía (la detecta lib/musica.py)
  - pista: "assets/musica/vibe_check.mp3"
    desde: s07          # fundido cruzado de 2 s al cambiar
```

**Marcas repetidas (actualizado).** Si la palabra de una marca aparece varias veces en la narración,
se indica cuál con `ocurrencia: 2`. Las marcas deben listarse en el orden en que se pronuncian.

### Reglas de validación (Etapa 1)

- Todos los campos obligatorios están presentes y `bloque` tiene un valor permitido.
- Cada frase de la narración tiene como máximo 25 palabras (en otro caso, aviso).
- Cada `palabra` de `marcas` aparece en su narración, **en el orden en que se listan** (si no, error:
  las animaciones llegarían tarde), y con `ocurrencia` cuando la palabra se repite (actualizado).
- La música existe y los segmentos citados en `desde`/`entrada` son del guion (actualizado).
- Cada componente listado existe en `lib/componentes/`, o se marca como "nuevo" para que Claude Code lo cree.
- La duración estimada (≈ 14 caracteres por segundo) cae dentro de ±20 % de `duracion_objetivo_min`.

---

## 10. Etapas del pipeline en detalle

### 10.1 Orquestador (`produce.py`)

```bash
python pipeline/produce.py --ep 00                 # pipeline completo
python pipeline/produce.py --ep 00 --desde e4      # relanzar desde el render
python pipeline/produce.py --ep 00 --solo e2 --segmento s03   # regenerar una voz
python pipeline/produce.py --ep 00 --borrador      # render rápido -ql para revisar
```

Cada etapa se ejecuta con `subprocess`, registra su log en `build/logs/` y devuelve un código de salida. El orquestador se detiene ante el primer fallo.

### 10.2 Diccionario de pronunciación

Los TTS en español pronuncian mal muchos términos técnicos. Antes de sintetizar, cada frase pasa por `config/pronunciacion.yaml`:

```yaml
# término: escritura fonética para el TTS
backpropagation: "bak-propaguéishon"
ReLU: "rélu"
softmax: "sóftmax"
PyTorch: "páitorch"
GAN: "gan"
GNN: "ge ene ene"
LSTM: "ele ese te eme"
GRU: "ge erre u"
MLP: "eme ele pe"
Transformer: "transfórmer"
embedding: "embéding"
dataset: "déitaset"
```

- La sustitución **solo afecta al texto que se envía al TTS**. Los subtítulos usan siempre el texto original.
- El diccionario crece episodio a episodio. Cada error detectado en QA se convierte en una entrada nueva.

### 10.3 Voz (Etapa 2)

**Creación de la voz de referencia (se hace una sola vez):**

1. Genera con Kokoro, usando una de sus voces en español, unos 15–20 segundos de texto neutro y bien articulado.
2. Guárdalo en `assets/voz_referencia/narrador.wav`.
3. Chatterbox clona esa voz en cada generación. El resultado es una voz propia del canal, consistente y sin problemas de derechos. **No se clona la voz de ninguna persona real sin su permiso.**
4. Prueba 10 frases técnicas con 3–4 configuraciones de parámetros y fija la mejor en `config/voz.yaml`.

```yaml
# config/voz.yaml (actualizado 2026-09-20)
motor: qwen3_tts                     # o chatterbox_multilingual
idioma: es
python: .venv-qwen/bin/python        # la etapa 2 se ejecuta con el venv de los modelos generativos
referencia: assets/voz_referencia/narrador.wav
texto_referencia: "..."              # Qwen lo necesita; lo rellena `python -m lib.voz_referencia`
semilla: 42
pausa_entre_frases_s: 0.25
sample_rate_salida: 48000
erre_inicial: false                  # r → rr en el texto del TTS (era por el acento de Chatterbox)
qwen:
  modelo: Qwen/Qwen3-TTS-12Hz-1.7B-Base
  idioma: Spanish
chatterbox:
  parametros: {exaggeration: 0.3, cfg_weight: 0.3, temperature: 0.7}
```

**Proceso por segmento:**

1. Dividir la narración en frases y procesar las pausas explícitas (`[pausa:0.4]`).
2. Aplicar el diccionario de pronunciación.
3. Sintetizar cada frase por separado.
4. Recortar los silencios del principio y del final de cada frase (filtro `silenceremove` de ffmpeg).
5. Concatenar las frases con la pausa configurada, remuestrear a 48 kHz y normalizar el volumen.
6. Guardar `build/audio/s01.wav` y `build/audio/s01.json`.

Esquema orientativo con Chatterbox Multilingual (la API puede variar entre versiones; verifícala en el repositorio oficial):

```python
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
wav = modelo.generate(
    texto_fonetico,
    language_id="es",
    audio_prompt_path="assets/voz_referencia/narrador.wav",
    exaggeration=0.4,
    cfg_weight=0.5,
)
ta.save("build/audio/frases/s01_f01.wav", wav, modelo.sr)
```

**Metadatos por segmento (`s01.json`):**

```json
{
  "segmento": "s01",
  "audio": "build/audio/s01.wav",
  "duracion": 9.84,
  "frases": [
    {"texto": "Mira estos puntos.", "inicio": 0.00, "fin": 1.12},
    {"texto": "Ninguna línea recta puede separarlos.", "inicio": 1.37, "fin": 3.90}
  ],
  "marcas": {"frontera_empieza": 7.62},
  "qa": {"wer": 0.04, "intentos": 1, "estado": "ok"}
}
```

**QA automático de la voz (actualizado tras construirlo):**

- **Transcripción con faster-whisper** (`language="es"`, `word_timestamps=True`) y cálculo del WER
  frente al texto original, excluyendo los términos del diccionario **y sus escrituras fonéticas**
  (Whisper puede transcribir cualquiera de las dos), con cifras convertidas a palabras en ambos lados.
- **Ritmo**: un umbral fijo de segundos por carácter no sirve, porque una frase larga se dice más
  rápido por carácter. Se compara con un modelo lineal del narrador (`duración ≈ a + b·caracteres`,
  r² ≈ 0,95) ajustado **por motor** con `python -m lib.ritmo`, y se acepta una franja de ratios.
- **Vocalizaciones**: si Whisper oye un "oh"/"eh" pegado a la voz, el intento se descarta (recortarlo
  deja restos audibles). Los sonidos separados por un silencio sí se recortan.
- **Si una frase falla**, se regenera con otra semilla (hasta `intentos_max`, hoy 5). Si sigue
  fallando, queda marcada y el pipeline se detiene. Tras escucharla: `--aceptar s01:f02` la da por
  buena y `--regenerar s01:f02` prueba semillas nuevas.
- **Caché por frase** con el motor, sus ajustes y la huella de la referencia: cambiar de voz
  regenera solo lo necesario, y cambiar un umbral **no** obliga a regenerar (se reevalúa).
- **Las marcas** salen de los tiempos por palabra de Whisper; si no la reconoce, se estima por la
  posición en la frase y se avisa.

### 10.4 Escenas sincronizadas (Etapa 3)

Todas las escenas heredan de una clase base que carga el audio y los tiempos de su segmento. Esquema:

```python
# lib/escena_narrada.py
import json
from manim import Scene

class EscenaNarrada(Scene):
    segmento: str = ""        # lo define cada escena, p. ej. "s01"
    episodio_dir: str = ""

    def setup(self):
        with open(f"{self.episodio_dir}/build/audio/{self.segmento}.json") as f:
            self.meta = json.load(f)
        self.add_sound(self.meta["audio"])

    @property
    def t(self) -> float:
        return self.renderer.time

    def hasta_marca(self, evento: str):
        """Espera hasta el instante en que la voz pronuncia la marca."""
        restante = self.meta["marcas"][evento] - self.t
        if restante > 0:
            self.wait(restante)

    def completar(self):
        """Rellena hasta el final del audio para que vídeo y voz midan lo mismo."""
        restante = self.meta["duracion"] - self.t
        if restante > 0:
            self.wait(restante)
```

Uso en una escena:

```python
class GanchoFrontera(EscenaNarrada):
    segmento = "s01"
    episodio_dir = "episodios/ep00_mlp"

    def construct(self):
        puntos = NubePuntos.lunas()
        self.play(FadeIn(puntos), run_time=1.5)
        recta = FronteraDecision.recta()
        self.play(Create(recta))
        self.hasta_marca("frontera_empieza")
        self.play(recta.animate.curvar_hacia(puntos), run_time=3)
        self.completar()
```

**Alternativa:** el plugin `manim-voiceover` resuelve algo parecido y admite motores de TTS personalizados. Se prefiere la clase propia porque evita depender de un plugin cuya compatibilidad con versiones recientes de Manim puede ir con retraso, y porque integra el QA de voz del pipeline.

**Generación de escenas con Claude Code:** para cada episodio, Claude Code lee `guion.yaml` (los campos `visual`, `componentes` y `marcas`) y escribe `escenas.py` reutilizando `lib/`. Se puede lanzar de forma no interactiva desde el orquestador:

```bash
claude -p "Lee episodios/ep00_mlp/guion.yaml y genera episodios/ep00_mlp/escenas.py siguiendo CLAUDE.md. Renderiza en -ql, extrae el último fotograma de cada escena, revísalo y corrige solapamientos."
```

### 10.5 Escenas de demo automatizadas

En vez de grabar la pantalla con OBS, que no se puede automatizar, las demos son **escenas de Manim construidas con datos reales**:

1. El script de `demo/` entrena el modelo en la RTX 5090 y guarda los logs: métricas por época, predicciones, imágenes generadas y fronteras de decisión.
2. La escena de demo muestra el fragmento de código relevante (componente `BloqueCodigo`, con resaltado de sintaxis) y anima los resultados reales: curvas de pérdida, rejillas de imágenes y mapas de atención.
3. Resultado: estética coherente con el resto del vídeo, datos 100 % reales y cero intervención manual.

Las demos se ejecutan **antes** del render, porque sus logs son una entrada de las escenas.

### 10.6 Render (Etapa 4)

```bash
manim -qh episodios/ep00_mlp/escenas.py GanchoFrontera -o s01.mp4 --media_dir episodios/ep00_mlp/build/media
```

- **Paralelización:** un proceso por segmento, con un máximo igual al número de núcleos físicos menos uno.
- **Modo borrador** (`--borrador`): se renderiza en `-ql` (480p a 15 fps) para iterar rápido.
- **Caché:** si ni `escenas.py` ni `lib/` ni el estilo ni el audio del segmento han cambiado, ese
  segmento no se vuelve a renderizar (actualizado).
- **Control de desfase (actualizado):** el vídeo de cada segmento no puede durar **más** que su voz
  (invadiría el siguiente); quedarse corto es inofensivo porque e6 rellena. Manim redondea cada
  animación a fotogramas completos, así que `EscenaNarrada` calcula las duraciones en fotogramas y
  redondea hacia abajo.

### 10.7 Hoja de contactos (Etapa 5)

Para cada segmento se extraen 3 fotogramas (al inicio, en la mitad y al final) y se componen en una sola imagen por episodio, con el `id` de cada segmento como etiqueta:

```bash
ffmpeg -ss 00:00:04.9 -i build/render/s01.mp4 -frames:v 1 build/contactos/s01_mitad.png
```

La composición se hace con Pillow. Esta imagen es el **punto de control 2**: en 2–3 minutos se detectan textos solapados, elementos fuera de cuadro o colores inconsistentes.

### 10.8 Ensamblado (Etapa 6)

**Concatenación** (todos los segmentos comparten resolución, fps y códec):

```bash
ffmpeg -f concat -safe 0 -i build/lista_segmentos.txt -c copy build/concat.mp4
```

**Música con ducking, normalización y codificación NVENC:**

```bash
ffmpeg -i build/concat.mp4 -stream_loop -1 -i assets/musica/fondo_01.mp3 \
  -filter_complex "\
    [0:a]asplit=2[voz][sc];\
    [1:a]volume=0.25[mus];\
    [mus][sc]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[musd];\
    [voz][musd]amix=inputs=2:duration=first:normalize=0,\
    loudnorm=I=-14:TP=-1.5:LRA=11[a]" \
  -map 0:v -map "[a]" \
  -c:v h264_nvenc -preset p7 -cq 19 -b:v 0 \
  -c:a aac -b:a 192k -ar 48000 \
  salida/ep00/ep00_mlp.mp4
```

- `sidechaincompress` baja automáticamente la música mientras suena la voz. Con `ratio=10` quedaba
  inaudible (37 dB bajo la voz); en `config/pipeline.yaml` se usa 4 (actualizado).
- **`loudnorm` no sirve aquí (actualizado).** Los picos de la voz impiden subir +6 dB en modo lineal,
  cambia a modo dinámico y se queda ~1 LU corto. En su lugar: ganancia + limitador de picos,
  midiendo y corrigiendo hasta quedar a ±0,1 LU de −14 LUFS. La conversión a estéreo va **antes**
  del limitador (ffmpeg baja 3 dB cada canal al duplicar el mono).
- **La voz manda:** se concatenan los .wav originales de cada segmento y cada vídeo se ajusta a un
  número exacto de fotogramas calculado sobre el tiempo acumulado (actualizado).
- **Música por tramos** con fundido cruzado, nivel igualado por sonoridad entre pistas y transición
  alineada con el cambio de bloque (`lib/musica.py` analiza tempo, subidas y secciones).
- Si el vídeo de Manim ya tiene buena calidad, se puede usar `-c:v copy` y ahorrarse la recodificación.
- Opcional: una intro y una outro fijas de 3–5 s, renderizadas una sola vez y reutilizadas en todos los episodios.

### 10.9 Subtítulos y capítulos (Etapa 7)

**Subtítulos:** como el audio se genera frase a frase, se conoce el tiempo exacto de cada frase. El `.srt` se construye con el **texto original del guion** (no con la transcripción de Whisper) y los tiempos de los JSON, sumando el desfase acumulado de cada segmento.

- El `.srt` se sube a YouTube como archivo aparte, en lugar de incrustarlo en la imagen. Es más accesible, se puede traducir automáticamente y ayuda al posicionamiento en búsquedas.
- Líneas de 42 caracteres como máximo; las frases largas se dividen.

**Capítulos:** se generan a partir del campo `bloque` y de las duraciones:

```
0:00 El problema
0:31 Por qué una recta no basta
1:34 La neurona, pieza a pieza
...
```

YouTube exige que el primer capítulo empiece en `0:00`, que haya al menos 3 y que cada uno dure al menos 10 segundos. Los títulos de capítulo salen de un campo opcional `titulo_capitulo` del primer segmento de cada bloque.

### 10.10 QA final (Etapa 8)

| Comprobación | Herramienta | Umbral orientativo |
|---|---|---|
| Duración del vídeo frente al audio | ffprobe | Diferencia < 0,1 s por segmento |
| Pantallas negras | `blackdetect` | Umbral bajo, ajustado al color de fondo; > 1 s = fallo |
| Imagen congelada | `freezedetect` | > 8 s = aviso (puede ser intencionado) |
| Silencios largos | `silencedetect=n=-45dB:d=2` | > 2 s = aviso |
| Sonoridad integrada | `loudnorm` en modo medición | −14 ± 1 LUFS |
| Duración total | ffprobe | Dentro de ±20 % del objetivo |

```bash
ffmpeg -i salida/ep00/ep00_mlp.mp4 -vf "blackdetect=d=1:pix_th=0.03" -an -f null - 2> build/qa/black.log
ffmpeg -i salida/ep00/ep00_mlp.mp4 -af "silencedetect=n=-45dB:d=2" -vn -f null - 2> build/qa/silencio.log
```

El resultado se guarda en `salida/ep00/informe_qa.md`, con estado OK, AVISO o FALLO para cada comprobación.

### 10.11 Miniatura y metadatos (Etapa 9)

- **Miniatura:** se envía un flujo de ComfyUI exportado "en formato API" a su servidor local (`http://127.0.0.1:8188/prompt`) con FLUX.1 [schnell]. El fondo se genera, y el título y el diagrama clave se superponen con Pillow usando la tipografía y la paleta del canal, para que las miniaturas sean reconocibles como serie. Se generan 3 variantes y eliges una.
- **Metadatos** (`metadatos.md`): título, descripción con capítulos, enlaces al código de las demos, atribuciones de música, nota de voz sintética (ver §17) y etiquetas.
- **Subida:** se recomienda **manual**, porque son unos 5 minutos. La API de YouTube Data es gratuita, pero una subida cuesta 1 600 de las 10 000 unidades de cuota diaria, y los proyectos de API no verificados pueden ver sus vídeos restringidos a privado. Si quieres automatizarla, revisa antes las condiciones vigentes.

---

## 11. Librería de componentes Manim

La librería `lib/componentes/` es la inversión que más rentabiliza la serie. Cada componente es una clase de Manim con parámetros, colores tomados de `lib/estilo.py` y animaciones propias.

| Componente | Descripción | Episodios |
|---|---|---|
| `Neurona` | Círculo con entradas, pesos visibles, sumatorio y activación | 00, 02, 07 |
| `Capa` / `RedDensa` | Columnas de neuronas con conexiones; `animar_forward()` y `animar_backward()` | 00, 01, 04, 05 |
| `FlechaGradiente` | Flecha naranja con magnitud visible; se desvanece si el gradiente es pequeño | 00, 02 |
| `PaisajePerdida` | Superficie 3D con una bola que desciende | 00 |
| `Matriz` | Rejilla de valores con resaltado de celdas, filas y columnas | 01, 03 |
| `KernelConvolucion` | Ventana que se desliza sobre una `Matriz` y produce otra | 01 |
| `MapaAtencion` | Matriz de tokens × tokens con intensidad de color y líneas entre palabras | 03, 07 |
| `CeldaLSTM` | Cinta de celda con puertas animables | 02 |
| `CuelloBotella` | Diagrama encoder → latente → decoder | 04, 05, 06 |
| `EspacioLatente2D` | Nube de puntos proyectada, coloreada por clase | 04, 05, 07 |
| `Grafo` | Nodos y aristas con `animar_mensajes()` | 07 |
| `BloqueCodigo` | Código con resaltado y líneas destacables | Todos (demos) |
| `GraficaEntrenamiento` | Curvas de pérdida y métrica animadas desde logs reales | Todos (demos) |
| `TarjetaComparativa` | "Úsala cuando / evítala cuando" | Todos |

**Reglas de la librería:**

- Ningún componente define colores ni tipografías propios: todo sale de `lib/estilo.py`.
- Cada componente tiene una escena de prueba en `lib/pruebas/` que Claude Code renderiza y revisa.
- Se crean **bajo demanda**. El piloto solo necesita 4–5 componentes y el resto aparece con cada episodio.

---

## 12. Puntos de control humanos

| # | Momento | Qué revisas | Tiempo |
|---|---|---|---|
| 1 | Antes de la Etapa 2 | `guion.yaml`: exactitud técnica, claridad, que el gancho enganche, que no haya relleno | 20–40 min |
| 2 | Después de la Etapa 5 | Hoja de contactos y una escucha rápida a 1,5× de los segmentos marcados por el QA | 10–15 min |
| 3 | Antes de publicar | Visionado completo a 1,25×, informe de QA, miniatura y metadatos | 20–30 min |

**Por qué no se eliminan:** un error conceptual, una fórmula mal escrita o una pronunciación absurda en un canal técnico destrozan la credibilidad, y ningún QA automático los detecta todos. Estos controles son baratos comparados con el daño que evitan.

---

## 13. Configuración del entorno

### 13.1 Windows nativo o WSL2

Se recomienda **WSL2 (Ubuntu)**:

- Manim, LaTeX, ffmpeg y las librerías de TTS dan menos problemas en Linux.
- La GPU funciona en WSL2 con el **driver de NVIDIA instalado solo en Windows**. No se instala ningún driver dentro de WSL.
- Claude Code funciona bien en WSL2.
- ComfyUI puede correr en Windows o en WSL; el pipeline se comunica con él por HTTP.

### 13.2 Pasos

**Cómo quedó (actualizado 2026-09-20).** Dos entornos, no uno:

| Entorno | Python | Para qué |
|---|---|---|
| `.venv` | 3.13 | Pipeline: Manim, ffmpeg, PyTorch cu128, demos. Chatterbox (motor de voz anterior) fija aquí `transformers 5.2` y `diffusers 0.29` |
| `.venv-qwen` | 3.12 | Modelos generativos: Qwen3-TTS (voz), faster-whisper, diffusers 0.39 (Wan 2.2, FLUX). La etapa 2 se ejecuta con este intérprete |

No se crean más entornos: lo generativo va todo en el segundo. Instalación reproducible con
`env/setup_wsl.sh` y `env/setup_qwen.sh`, y versiones fijadas en `env/requirements.lock.txt`.
Cuidado al instalar en `.venv-qwen`: hay que fijar `huggingface-hub` y `transformers`, porque
`diffusers` los subiría y rompería qwen-tts.

```bash
# 1. Dependencias del sistema (Ubuntu en WSL2)
sudo apt update
sudo apt install -y build-essential python3-dev ffmpeg \
  libcairo2-dev libpango1.0-dev pkg-config \
  texlive texlive-latex-extra texlive-fonts-extra texlive-lang-spanish dvisvgm

# 2. Entorno Python (uv o venv)
python3 -m venv .venv && source .venv/bin/activate

# 3. PyTorch con soporte para Blackwell (CUDA 12.8 o superior)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Resto de dependencias
pip install manim faster-whisper chatterbox-tts kokoro pyyaml pillow jiwer
pip install torch_geometric   # para el episodio 07

# 5. Verificación
python env/verificar_entorno.py
```

> **Importante:** algunas librerías (TTS incluido) pueden intentar reinstalar una versión de PyTorch sin soporte para la 5090 al instalarse. Después de cada `pip install`, ejecuta de nuevo `verificar_entorno.py`. Si la versión de torch ha cambiado, reinstala la build `cu128` y fija las versiones en `pyproject.toml`.

### 13.3 Gestión de la VRAM (16 GB)

| Proceso | VRAM aproximada | ¿A la vez que otros? |
|---|---|---|
| Chatterbox Multilingual | Pocos GB | Sí, con Whisper |
| faster-whisper large-v3 (fp16) | ~4–5 GB | Sí, con el TTS |
| Demos de entrenamiento | Variable | Mejor en solitario |
| ComfyUI + FLUX.1 schnell (fp8) | Hasta ~12 GB o más | **Solo** |

Como cada etapa es un subproceso, la memoria se libera al terminar. ComfyUI se arranca solo durante la Etapa 9, o se cierra antes de lanzar el TTS.

---

## 14. Trabajo con Claude Code

### 14.1 Reparto de responsabilidades

| Tarea | Responsable |
|---|---|
| Construir el pipeline, la librería y los scripts de entorno | Claude Code (una vez) |
| Borrador de `guion.yaml` a partir de este plan | Claude Code, con tu revisión (control 1) |
| `escenas.py` de cada episodio | Claude Code |
| Código de las demos | Claude Code |
| TTS, render, montaje, subtítulos y QA | Pipeline automático en local |
| Decisiones editoriales y visto bueno final | Tú |

### 14.2 Orden de las sesiones

1. **Sesión 1 — Entorno** (en modo plan, con Shift+Tab): instalación, `verificar_entorno.py` y fijado de versiones.
2. **Sesión 2 — Esqueleto:** estructura del repo, `CLAUDE.md`, configuración YAML y `produce.py` con etapas vacías.
3. **Sesión 3 — Voz:** Etapa 2 completa, voz de referencia, diccionario de pronunciación y QA con Whisper. Prueba con 10 frases técnicas.
4. **Sesión 4 — Sincronización:** `EscenaNarrada` y un episodio de prueba de 3 segmentos triviales, de principio a fin.
5. **Sesión 5 — Ensamblado y QA:** Etapas 5 a 8.
6. **Sesión 6 — Librería:** los componentes que necesita el episodio 00.
7. **Sesión 7 en adelante — Episodios:** guion → control 1 → escenas → control 2 → final → control 3.

### 14.3 Cómo revisa Claude Code sus propias animaciones

Claude Code no puede ver vídeo, pero sí imágenes. En `CLAUDE.md` se le indica que:

- itere con `manim -ql` (baja calidad y rápido);
- exporte fotogramas clave (`manim -s` da el último fotograma, o ffmpeg con `-ss`) como PNG;
- los mire para detectar solapamientos, elementos fuera de cuadro, texto ilegible o colores fuera de la paleta;
- solo renderice en `-qh` cuando la revisión visual sea correcta.

### 14.4 Prompts útiles

- *"Lee `episodios/ep01_cnn/guion.yaml`. Lista los componentes que faltan en `lib/`, créalos con su escena de prueba y revisa sus fotogramas."*
- *"El QA marca el segmento s07 del episodio 02 con WER 0,31. Escucha la transcripción de Whisper frente al texto y propón entradas nuevas para `pronunciacion.yaml`."*
- *"Revisa `build/contactos/ep03.png` y lista los problemas visuales por segmento, sin corregir nada todavía."*

---

## 15. Hoja de ruta

**Estado a 2026-09-20:** fases 0 a 2 terminadas; fase 3 (piloto del episodio 00) en revisión.

| Fase | Semanas | Entregable | Criterio de salida |
|---|---|---|---|
| 0. Entorno | 1 | Entorno verificado y versiones fijadas | ✅ hecho |
| 1. Voz | 1 | Voz del canal definida y diccionario inicial | ✅ hecho (toma 2; motor cambiado a Qwen3-TTS el 20-09) |
| 2. Pipeline | 2 | Etapas 1 a 8 funcionando con un episodio de prueba | ✅ hecho (`ep99`, 3,7 s con todo en caché) |
| 3. Piloto | 1 | Versión corta del episodio 00 | 🔄 borrador de 3:41 montado; pendiente: voz algo rápida y animaciones por refinar |
| 4. Episodio 00 completo | 1 | Primer vídeo publicado | Pasa los 3 controles |
| 5. Serie | 1 episodio cada 1–2 semanas | Episodios 01 a 07 y el extra | Tiempo humano por episodio < 4 h a partir del tercero |
| 6. Retrospectiva | Tras 3 episodios | Ajustes basados en datos | Análisis de retención y CTR |

**Métricas que conviene seguir:**

- **Retención media y curva de retención.** Como cada caída se puede mapear a un segmento de `guion.yaml`, sabrás exactamente qué parte del guion reescribir.
- **CTR de la miniatura.**
- **Comentarios con dudas recurrentes.** Son candidatas a vídeos cortos o a mejoras del episodio.

---

## 16. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| PyTorch sin soporte para Blackwell tras instalar una dependencia | Alta | Alto | Build `cu128` fijada; `verificar_entorno.py` tras cada instalación |
| Pronunciaciones erróneas del TTS | Alta | Medio | Diccionario, QA con Whisper, regeneración automática y control 2 |
| Voz inconsistente entre episodios | Media | Alto | Referencia, semilla y parámetros fijos; mismas versiones del modelo |
| Errores visuales en las escenas generadas | Alta | Medio | Revisión por fotogramas de Claude Code, hoja de contactos y librería probada |
| Errores conceptuales en el guion | Media | Muy alto | Control 1 obligatorio; contrastar con fuentes de referencia |
| Render de Manim lento | Media | Bajo | Paralelización, caché por segmento y modo borrador |
| Falta de VRAM | Media | Bajo | Etapas como subprocesos; ComfyUI en solitario |
| Cambios de licencia en modelos | Baja | Alto | Copia de las licencias en `docs/licencias/`; motores alternativos identificados |
| Ruptura de APIs entre versiones de librerías | Media | Medio | Versiones fijadas; actualizar solo entre episodios |
| Contenido considerado "no auténtico" por YouTube | Media | Alto | Ver §17 |
| Quemarse o abandonar la serie | Media | Muy alto | Hoja de ruta realista; el piloto valida antes de invertir más |

---

## 17. Políticas de YouTube, ética y transparencia

- **Contenido no auténtico:** YouTube no monetiza el contenido producido en masa o repetitivo con poca aportación humana. La serie lo evita con guiones propios revisados, demos reales con código publicado y un enfoque editorial personal (qué arquitectura usar y cuándo no). **La automatización es de la producción, no del criterio.**
- **Divulgación de contenido sintético:** YouTube exige declarar el contenido realista alterado o generado con IA. Una narración sintética sobre animaciones educativas no suele encajar en el caso típico, pero las normas cambian: revísalas antes del primer vídeo y ante cualquier duda, marca la casilla.
- **Transparencia con la audiencia:** se recomienda añadir en la descripción una línea como *"Narración generada con voz sintética. Guion, animaciones y demos: [tu nombre/canal]"*. Genera más confianza que ocultarlo.
- **Voz:** la voz del canal es sintética y propia. No se clonan voces de personas reales sin su consentimiento explícito.
- **Datos:** solo datasets públicos con licencia compatible, o datos sintéticos. Nunca datos de clientes ni del trabajo.
- **Música:** respeta las condiciones de cada pista, incluidas las que exigen atribución en la descripción.

---

## 18. Checklist por episodio

**Preproducción**
- [ ] `guion.yaml` redactado y validado (Etapa 1)
- [ ] **Control 1:** guion revisado (exactitud técnica, gancho, puente)
- [ ] Términos nuevos añadidos a `pronunciacion.yaml`
- [ ] Demo ejecutada y logs generados

**Producción**
- [ ] Voz generada y QA de voz en OK
- [ ] Componentes nuevos creados con su escena de prueba
- [ ] `escenas.py` generado y revisado en borrador (`-ql`)
- [ ] Render en `-qh` completado
- [ ] **Control 2:** hoja de contactos y segmentos marcados revisados

**Postproducción**
- [ ] Ensamblado con música y sonoridad correctas
- [ ] `.srt` y capítulos generados
- [ ] Informe de QA sin FALLOS
- [ ] Miniatura elegida entre 3 variantes
- [ ] Metadatos completos (capítulos, atribuciones, nota de voz sintética, enlace al código)
- [ ] **Control 3:** visionado completo

**Publicación**
- [ ] Vídeo subido con su `.srt`
- [ ] Código de la demo publicado (por ejemplo, en GitHub)
- [ ] A los 7 días: revisar la curva de retención y anotar los segmentos débiles

---

## 19. Anexos

### Anexo A — Plantilla de `CLAUDE.md`

```markdown
# Serie de vídeos: arquitecturas de redes neuronales

## Objetivo
Pipeline automático local (RTX 5090, WSL2) que produce vídeos educativos en español
a partir de episodios/<ep>/guion.yaml. Orquestador: pipeline/produce.py.

## Reglas generales
- El guion (guion.yaml) es la única fuente de verdad. Subtítulos, capítulos y marcas se derivan de él.
- Cada etapa del pipeline es un script independiente que lee y escribe archivos. Nunca cargues dos modelos grandes en el mismo proceso.
- PyTorch debe ser la build cu128. Si instalas algo, ejecuta después env/verificar_entorno.py.
- Nunca uses datos de clientes ni del trabajo: solo datasets públicos o sintéticos.

## Manim
- Todas las escenas heredan de lib/escena_narrada.EscenaNarrada y se sincronizan con hasta_marca() y completar().
- Reutiliza siempre los componentes de lib/componentes/ antes de crear animaciones nuevas.
- Colores y tipografías: solo desde lib/estilo.py. Nunca uses valores hexadecimales en las escenas.
- Máximo 7 palabras por elemento de texto en pantalla. Una idea por pantalla.
- Itera con `manim -ql`. Tras cada cambio, exporta fotogramas clave a PNG y revísalos
  (solapamientos, elementos fuera de cuadro, texto ilegible, colores fuera de paleta).
  Renderiza en -qh solo cuando la revisión visual sea correcta.
- Cada componente nuevo lleva su escena de prueba en lib/pruebas/.

## Guiones
- Tuteo, frases de menos de 25 palabras, sin relleno.
- Estructura de 7 bloques: gancho, limitacion, intuicion, matematicas, demo, cuando_usar, puente.
- Los términos técnicos en inglés van también a config/pronunciacion.yaml.

## Qué NO hacer
- No edites los archivos de salida/ a mano.
- No cambies versiones de dependencias sin avisar.
- No metas vídeo generativo en un episodio sin decidirlo antes: está en evaluación (§20) y por
  ahora solo se usa como prueba con pipeline/wan_clip.py. Todo lo visual publicado es Manim.
```

### Anexo B — `env/verificar_entorno.py`

```python
"""Verifica que el entorno está listo para el pipeline en la RTX 5090."""
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
        resultados.append(check("GPU", True, f"{nombre} (compute capability {cap[0]}.{cap[1]})"))
        a = torch.randn(4096, 4096, device="cuda")
        b = (a @ a).sum().item()   # falla si la build no soporta Blackwell
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

for modulo in ["manim", "faster_whisper", "chatterbox", "yaml", "PIL", "jiwer"]:
    try:
        __import__(modulo)
        resultados.append(check(f"import {modulo}", True))
    except Exception as e:
        resultados.append(check(f"import {modulo}", False, repr(e)))

sys.exit(0 if all(resultados) else 1)
```

> Nota: el ffmpeg de los repositorios de Ubuntu puede no incluir NVENC dentro de WSL2. Si la comprobación falla, usa una build estática de ffmpeg con NVENC o codifica con `libx264` (más lento, pero con la misma calidad).

### Anexo C — `config/pipeline.yaml` (valores iniciales)

```yaml
render:
  calidad_final: qh          # 1080p60
  calidad_borrador: ql
  procesos_paralelos: auto   # núcleos físicos - 1

voz_qa:
  wer_max_frase: 0.15
  intentos_max: 3
  seg_por_caracter: [0.045, 0.095]

audio:
  lufs_objetivo: -14
  true_peak: -1.5
  volumen_musica: 0.25

qa_final:
  tolerancia_duracion_segmento_s: 0.1
  negro_max_s: 1.0
  congelado_aviso_s: 8
  silencio_aviso_s: 2
  tolerancia_duracion_total: 0.20

subtitulos:
  max_caracteres_linea: 42
  max_lineas: 2
```

### Anexo D — Primer prompt para Claude Code

```
Lee plan_serie_redes_neuronales.md completo. Vamos a empezar por la Fase 0 (entorno).
En modo plan, propón los pasos para dejar listo un entorno WSL2 con PyTorch cu128,
Manim con LaTeX, ffmpeg con NVENC, faster-whisper y Chatterbox Multilingual, y crea
env/verificar_entorno.py según el Anexo B. No instales nada hasta que apruebe el plan.
Después, crea la estructura de repositorio de la sección 8.3 y el CLAUDE.md del Anexo A.
```

---

## 20. Bitácora de decisiones y pendientes

### Decisiones tomadas durante la construcción (2026-09-19, 20 y 10-03)

| Decisión | Por qué |
|---|---|
| **Qwen3-TTS en vez de Chatterbox** | Comparados con las mismas frases y la misma voz de referencia: Qwen suena natural, sin el acento inglés en «red» ni los "oh" y colas que había que recortar. Las 58 frases del episodio 00 pasaron el QA a la primera |
| **Dos entornos, no uno** | Chatterbox fija `transformers 5.2`; Qwen y diffusers piden otras versiones. El segundo entorno aloja todo lo generativo |
| **El ritmo del narrador se modela, no se fija** | Un umbral de s/carácter marcaba como malas frases correctas: las largas se dicen más rápido por carácter |
| **Las marcas van en orden y admiten `ocurrencia`** | Cuatro marcas desordenadas hacían llegar tarde las animaciones; «hongo» aparecía dos veces y se ataba a la primera |
| **Ganancia + limitador en vez de `loudnorm`** | `loudnorm` se quedaba 1 LU corto por los picos de la voz |
| **Música por tramos, alineada con los bloques** | Una pista de 2:37 se repetía en un episodio de 4 min; y la entrada de la música refuerza el cambio de bloque |
| **Una marca por frase** | El piloto tenía 25 marcas para 58 frases: 223 de 263 s sin que cambiara nada en pantalla. Con 66 marcas bajó a 43 s |
| **Los pesos de las animaciones son reales** | `demo/entrenar_capas.py` entrena la red del episodio 00; con una capa de 2 neuronas el error se queda en 10 % y con dos capas baja a 0 %, que es justo el argumento del guion |
| **El ejemplo del episodio 00 es un invernadero** | «Mira estos puntos» no engancha; con dos sensores y un hongo, el ejemplo recorre el episodio entero |
| **Los Reels son canal propio, no promoción** | Identidad visual separada (dorado jerárquico frente a la paleta semántica de §6.1). Lo único que comparten con la serie larga es el narrador: es lo que hace reconocible al autor cuando cambia el formato |
| **La clase corta es la unidad de producción** | Las dos referencias de §21 son el mismo sistema: el vídeo horizontal de cinco minutos es un montaje de diez clases cortas. Se construye la clase una vez |
| **La voz manda sobre la duración de la escena** | En `reel_voz.TimelineNarrada` la escena dura lo que su frase y el dibujo se repite debajo, en vez de meter la animación en un número fijo de fotogramas. Las funciones de dibujo no cambian |
| **El bucle reinicia en seco, sin fundido** | Fundir el final de una vuelta con el principio de la siguiente superpone los números de la leyenda y parece un fallo de render |
| **WSL con 48 GB** | Estaba limitado a 24 de los 63,4 físicos y el OOM killer mataba la carga de Wan (22 GB de pesos en float32) |

### Pendientes, por orden

1. **Velocidad de la voz: decidir.** `velocidad: 0.92` ya está implementado (se aplica con
   ffmpeg `atempo` antes del QA, para que lo medido sea lo publicado) pero **las frases del
   episodio 00 no se han regenerado todavía**, así que nunca se ha escuchado. Y según §21.5 deja la
   narración en 170 ppm, por debajo de las dos referencias y de los 184 ppm actuales. Regenerar,
   escuchar, decidir, y recalcular el modelo de ritmo con `python -m lib.ritmo`.
2. **Animaciones más finas.** Curvas de aceleración, movimientos de cámara (zoom y paneo),
   apariciones escalonadas, profundidad y transiciones entre escenas, en vez de cortes secos.
   Es trabajo de oficio en Manim y afecta a `lib/componentes/` y a `lib/estilo.py`.
3. **Etapa 9: miniatura y metadatos**, con FLUX.2 [klein] 4B (Apache 2.0).
4. **Vídeo generativo (Wan 2.2), en evaluación.** Herramienta suelta en `pipeline/wan_clip.py`.
   Criterio para adoptarlo: que un clip en el gancho mejore la retención frente a la versión solo
   con Manim. Reglas si se adopta: nada de texto generado, nada de personas realistas, un modelo
   cada vez en la GPU y una línea en la descripción declarándolo.
5. **Efectos de sonido puntuales** (campo `sfx` en las marcas y mezcla en e6): dos o tres por
   episodio como máximo, después del piloto.
6. **Demo real del episodio 00** (MLP frente a gradient boosting en datos tabulares).
7. **Regenerar los siete Reels en esta máquina** (`redes_neuronales_vscode/run_reels.sh`, con
   `PATH=../.venv/bin:$PATH`). El `.venv` de la serie ya cubre sus dependencias; no hace falta
   entorno nuevo.
8. **Módulo `clase.py`** con la anatomía de §21.2 sobre `reel_common` + `reel_voz`: cabecera,
   panel, barra de capítulo con procedencia y panel de código con línea resaltada.
9. **Edición didáctica de 16 líneas** por episodio (§21.6). Empezar por `train_gnn.py` (26 líneas)
   o `train_dbn.py` (28), que son los que menos hay que destilar.


---

## 21. Referencias de calidad y formato de clase corta

Dos vídeos del canal **@machinelearningtogo** fijan el listón, y están analizados fotograma a
fotograma (ambos en `series-redes-neuronales/`, fuera del repositorio publicable):

| | Vertical · *XGBoost* | Horizontal · *ML explained in 5 minutes* |
|---|---|---|
| Lienzo | 720×1280 a 30 fps | 16:9 (la copia analizada, 640×360) |
| Duración | 1:52 · 336 palabras | 5:10 · 880 palabras |
| Ritmo | 180 palabras/min | 170 palabras/min |
| Alcance | un tema a fondo | diez ideas encadenadas |
| Reparto | panel animado arriba, código abajo | código a la izquierda, animación a la derecha |
| Navegación | barra de capítulo con **procedencia** | barra con los diez capítulos, el actual resaltado |

### 21.1 La unidad de producción es la clase de dos minutos

El horizontal **no es otro producto: es un montaje de diez clases cortas**, cada una con su propio
archivo de código (`linear_regression.py`, `one_neuron.py`, `act.py`, `gd.py`, `bp.py`, `see.py`,
`overfitting.py`, `jobs.py`) y su pie de plano (`w -= lr * dw`, `the blame travels backwards`,
`the error as a landscape`). Cierra con una pantalla resumen: las diez miniaturas más dos preguntas,
*«¿qué es el aprendizaje automático?»* y *«¿cómo aprende?»*.

Consecuencia para la producción: **se construye la clase corta una vez**. El vídeo largo en
horizontal se obtiene encadenando clases y cambiando el reparto de pantalla, no rodando otra cosa.
Prioridad: la clase de dos minutos (Reel largo y Short de YouTube); el horizontal es su extensión.

### 21.2 Anatomía del formato vertical (medidas reales)

| Franja | y | Contenido |
|---|---|---|
| Cabecera | 0–130 | marca en rojo + «PART 22», título del tema y **la tesis en una línea** |
| Panel | 140–640 | lo único que cambia |
| Barra de capítulo | ~660 | izquierda: el compás, en azul y minúsculas · derecha: de dónde sale |
| Código | 693–1130 | 16 líneas fijas, con la línea activa resaltada |

### 21.3 Reglas narrativas

1. **El gancho es una pregunta, en la primera frase.** Sin presentación ni intro.
2. **La tesis no sale nunca de pantalla** (subtítulo de cabecera) y es también la penúltima frase.
   Abrir y cerrar con la misma idea.
3. **Continuidad explícita entre episodios**, escrita en el cuadro: «part 18, same data»,
   «part 19: weight -= lr * slope». No basta con decirlo.
4. **Todo son números concretos**: 47 %, 400 clientes, 200 árboles, 84 de 100, 16 líneas.
   Nada de «muy preciso» ni «bastante bueno».
5. **Alcance honesto en el cierre**: «gana en datos tabulares», no «es el mejor modelo».
6. **Una idea por compás**, con su nombre en la barra: `a forest votes`, `a small step: 1%`,
   `every tree is one step`.
7. Llamada a la acción de una sola frase, al final.

### 21.4 Reglas técnicas

1. **Ancla fija.** El código no se va de la pantalla. Lo que cambia es el panel. El ojo siempre
   tiene dónde volver: es lo que hace que no agobie aunque pase algo cada cuatro segundos.
2. **Sin cortes duros.** La detección de planos sobre el vertical da **cero** cambios de plano en
   112 segundos: todo se transforma, nada corta.
3. **Procedencia en la barra**: `printed by line 14`, `the loop`, `347 of 400 closer`,
   `tree 147 · new: 84 right`, `guess 90 %, tree 187`. Cada imagen declara de qué línea de código o
   de qué número del entrenamiento sale. Es lo que separa «mira qué bonito» de «mira qué hace esto».
4. **Anotación dibujada a mano**, en naranja, sobre el código y sobre los datos: círculos que
   rodean `range(200)` o un grupo de puntos. Da el gesto de pizarra sin romper el diseño.
5. **Los números salen del modelo real.** El árbol que enseña tiene umbrales de verdad
   («income above 3.020?») y hojas de verdad (−0,39, −0,07, +0,47, −0,20). Coincide con el principio
   de esta serie: los pesos de las animaciones son reales (§20).
6. **Color jerárquico, no semántico** (al contrario que §6.1): dorado = la idea clave,
   azul = los datos, verde/rojo = acierto/fallo. En un vídeo de dos minutos el color ordena la
   atención; en uno de doce, significa. No se unifican las dos paletas, pero **un mismo color no
   puede significar cosas opuestas** entre formatos.

### 21.5 El ritmo no era el problema

Medido sobre el audio real de los tres:

| | palabras/min |
|---|---|
| Referencia vertical | 180 |
| Referencia horizontal | 170 |
| Episodio 00 de esta serie (60 frases, 214 s) | 184 |

Hablan igual de rápido. Lo que hace que las referencias no se sientan aceleradas es el **ancla**,
no la lentitud. El ajuste `velocidad: 0.92` deja la narración en 170 ppm, por debajo de las dos
referencias: queda pendiente decidir si se mantiene (ver §20).

Confirmado al generar la primera demo narrada (`notebooks/05_voz_en_reels.ipynb`): con
`velocidad: 0.92` el narrador habla a **168 ppm** medidos.

**Sesgo del modelo de ritmo con frases cortas.** En esa misma demo, `duracion_modelo` se quedó
**un 13 % corto** (19,6 s previstos frente a 22,1 s reales), y hasta un 30 % en la frase más corta
(3,4 s previstos, 4,4 s reales). El modelo se ajustó sobre las frases de un episodio largo, donde
pesan poco los arranques y los finales. Al presupuestar un guion de clase corta, **añade un 15 %**
a lo que diga la predicción, o reajusta el modelo incluyendo frases de menos de 60 caracteres.

### 21.6 La edición didáctica de 16 líneas

El panel de código de la referencia no es el script de entrenamiento: es un archivo escrito para
caber en pantalla, que sigue siendo código real y ejecutable. Los entrenamientos de este proyecto
van de 26 a 74 líneas:

```
train_gnn.py 26 · train_dbn.py 28 · train_dbn_ola.py 39 · train_gnn_es.py 48
train_pueblos.py 64 · train_paint.py 64 · train_cuadros.py 74
```

Cada episodio necesita, por tanto, **su edición didáctica de 16 líneas** como entregable aparte.
No es trabajo mecánico: es decidir qué es el algoritmo y qué es andamiaje.

### 21.7 Qué no se copia

La marca, el idioma y la voz son propios. De las referencias se toman la **anatomía**, las reglas
narrativas y el nivel de acabado, no la identidad.

---

*Documento vivo: actualízalo tras la retrospectiva de la Fase 6 y cada vez que cambie una decisión de stack.*
