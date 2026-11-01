# Serie «Redes Neuronales» · proyecto completo (todas las versiones)

Shorts/Reels verticales (1080×1920, 30 fps, 18–23 s) donde **una red neuronal real se entrena delante del espectador**.
Todo en **NumPy a mano** (backprop, Adam, LSTM, atención, GAN, GCN, RBM…), sin frameworks.

El proyecto contiene **tres generaciones** de cada episodio:

| Generación | Qué es | Scripts | Tipografía |
|---|---|---|---|
| **Clásica** | primeros prototipos (dígitos, espirales, kárate…) | `train_*.py` + `render_*.py` | Space Grotesk + JetBrains Mono |
| **«b»** | mismos episodios con datos cercanos (cuadros, pueblos) en formato panel | `train_*.py` + `render_*.py` | Space Grotesk + JetBrains Mono |
| **Reels** ⭐ | formato Instagram: gancho, escenas cortas, zonas seguras, curvas legibles | `reel_*.py` (+ `reel_common.py`) | **Archivo Black + Inter** |

## 1. Arranque en VS Code

```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # + ffmpeg instalado en el PATH
mkdir -p work/hd out
./run_reels.sh        # entrena lo que falte y genera todos los Reels en out/
./run_all.sh          # (opcional) versiones clásicas y «b» en formato panel
```

En la pestaña **Run and Debug** de VS Code hay una configuración por Reel para **previsualizar fotogramas** sin renderizar
el vídeo (se guardan como PNG en `work/`). Requiere haber entrenado ese episodio antes.

### Comandos de los Reels (todos iguales gracias a `reel_common.cli`)

```bash
python3 reel_gnn.py prev 40 300 470     # PNG de esos fotogramas en work/
python3 reel_gnn.py chunk 1e12          # todos los fotogramas a work/frames_<nombre>/ (reanudable: salta los que existen)
python3 reel_gnn.py encode              # audio + vídeo -> out/<nombre>.mp4
```
Excepción: `reel_paint.py` lleva delante la tipografía: `python3 reel_paint.py C prev 300` (A y B = tipografías alternativas).

## 2. Reels (versión recomendada para publicar)

| Ep. | Reel | Entrenamiento | Datos | Salida |
|---|---|---|---|---|
| 1b | «¿Puede una red pintar un cuadro?» red normal vs Fourier, curva de dB, zoom ×8 | `train_paint.py 6000 4096` → `precompute_hd2.py` | *La noche estrellada* | `ep1b_reel_fuente_C.mp4` |
| 2b | «¿De qué cuadro es este trozo?» + giro sin color | `train_cuadros.py 800 rgb` y `2000 gray` | 5 cuadros DP | `ep2b_reel_cuadros.mp4` |
| 3b | «Uno de estos pueblos no existe» + curva de error | `train_pueblos.py 3500` → `select_pueblos.py` | 7.665 municipios INE | `ep3b_reel_pueblos.mp4` |
| 4 | *pendiente*: «¿Puede una red poner las tildes?» | — | palabras del español | — |
| 5b | «¿Cabe un cuadro en 2 números?» + paseo latente | `train_ae_cuadros.py 4000` | 5 cuadros DP | `ep5b_reel_autoencoder.mp4` |
| 6 | «Estos pueblos no existen»: GAN que dibuja España, *mode collapse*, reto A/B | `train_gan_es.py` (reanudable) | 8.042 municipios con coordenadas | `ep6_reel_gan_espana.mp4` |
| 7 | «¿De qué comunidad es este pueblo?»: nombre (30,6 %) vs nombre+vecinos (93 %), enclaves | `train_gnn_es.py` | 8.130 municipios + grafo k=8 | `ep7_reel_gnn_comunidades.mp4` |
| 8b | «40 % de píxeles rotos»: aprende soñando y restaura | `train_dbn_ola.py` | *La gran ola* binarizada | `ep8b_reel_ola.mp4` |

### `reel_common.py` (la base de todos los Reels)
Tipografía C, paleta, fondo, `headline()`, `body()`, `chip()`, `paste_img()` (esquinas redondeadas), `shade()`
(transparencias bien mezcladas sobre RGBA), `chart()` (gráfica de pérdida legible en móvil: líneas con halo, valor actual
en la leyenda, eje log opcional), `Timeline` (escenas + «punch-in» en cada corte), `Audio` (pad, whoosh, tonos, campanada) y `cli()`.

**Reglas de diseño** (respetarlas en episodios nuevos):
- Contenido importante entre **y≈260 e y≈1480** y fuera de la franja derecha baja: Instagram tapa el resto.
- Titulares en mayúsculas, máx. 2 líneas y ~6 palabras por línea; la última línea en dorado = la idea clave.
- Explicaciones en Inter, minúsculas, ≥ 32 px. Etiquetas técnicas sin forzar mayúsculas cuando cambian el significado (`(x, y)`).
- Gancho = el resultado o un reto en el primer segundo. Cambio visual cada 2–4 s. El último plano enlaza con el primero si se puede.
- Cada episodio enseña **la curva de entrenamiento** de su problema.

## 2bis. Episodios largos (2–3 min) · carpeta `explainer/` ⭐ Temporada 1 completa (8 episodios)

Formato educativo: un episodio por arquitectura, con **intuición animada, fórmulas, código y demo real**, narrado con la voz
del canal y con subtítulos sincronizados. Marco fijo de la serie (navegación de las 8 arquitecturas, barra de progreso,
título, sección, panel, tarjeta «usos / ojo» y subtítulos).

| Archivo | Qué es |
|---|---|
| `explainer/explainer.py` | Motor: marco fijo, `Formula` (LaTeX vía matplotlib, término a término), `CodeBlock` (Pygments, resaltado por líneas), `Timeline` dirigida por la voz, subtítulos con palabras clave en dorado, audio y CLI |
| `explainer/guion_ep01_mlp.yaml` | Guion por segmentos: narración, escena, sección y **cues** (la palabra que dispara cada animación) |
| `explainer/ep01_mlp.py` | Episodio 01 · MLP (11 escenas, 2:56) |
| `explainer/ep02_cnn.py` + `guion_ep02_cnn.yaml` | Episodio 02 · CNN (11 escenas, 2:45). Requiere `work/run_cuadros_rgb.pkl` y `work/run_cuadros_gray.pkl` |
| `explainer/ep03_rnn.py` + `guion_ep03_rnn.yaml` | Episodio 03 · RNN/LSTM (10 escenas, 2:35). Requiere `work/run_pueblos.pkl` y `data/pueblos_sel.json` |
| `explainer/ep04_transformer.py` + `guion_ep04_transformer.yaml` | Episodio 04 · Transformers: la red que pone tildes (10 escenas, 2:44). Requiere `work/run_tildes.pkl`: `python3 train_tildes.py` (reanudable; ~6 min en 1 núcleo). Datos: `data/tildes_data.json` (regenerable con `python3 data_tildes.py`, necesita `wordfreq`) |
| `explainer/ep05_ae.py` + `guion_ep05_ae.yaml` | Episodio 05 · Autoencoders (10 escenas, 2:37). Requiere `work/run_ae_cuadros.pkl` (`python3 train_ae_cuadros.py 4000`) |
| `explainer/ep06_gan.py` + `guion_ep06_gan.yaml` | Episodio 06 · GAN (10 escenas, 2:45). Requiere `work/run_gan_es.pkl` (`python3 train_gan_es.py`, reanudable) |
| `explainer/ep07_gnn.py` + `guion_ep07_gnn.yaml` | Episodio 07 · Redes de grafos (10 escenas, 2:48). Requiere `work/run_gnn_es.pkl` (`python3 train_gnn_es.py`) |
| `explainer/ep08_dbn.py` + `guion_ep08_dbn.yaml` | Episodio 08 · DBN, cierre de temporada (11 escenas, 2:30). Requiere `work/run_dbn_ola_v2.pkl` (`python3 train_dbn_ola.py`) |
| `explainer/tts.py` | Voz con Chatterbox Multilingual + tiempos por palabra con faster-whisper → `explainer/voz/` |

```bash
# desde la raíz del proyecto; requiere work/run.pkl (python3 train.py) y work/hd_hero.npy (python3 precompute_hd2.py)
python3 explainer/ep01_mlp.py info                    # duración por segmento y total
python3 explainer/ep01_mlp.py prev 300 1400           # fotogramas sueltos a work/
python3 explainer/ep01_mlp.py chunk 1e12              # todos los fotogramas (reanudable)
python3 explainer/ep01_mlp.py encode                  # -> out/ep01_mlp_piloto.mp4

# con la voz (en la máquina con GPU; ver requirements-voz.txt)
python3 explainer/tts.py explainer/guion_ep01_mlp.yaml --ref ruta/voz_referencia.wav --seed 42
rm -rf work/frames_ep01_mlp_piloto && python3 explainer/ep01_mlp.py chunk 1e12 && python3 explainer/ep01_mlp.py encode
```

**Cómo se sincroniza:** sin voz, cada segmento dura `palabras / 2,4 s` y las palabras se reparten de forma uniforme.
Con voz, `tts.py` escribe `explainer/voz/<guion>/timings.json` (una carpeta por episodio) con la duración real de cada segmento y el instante de cada palabra
(faster-whisper + alineación con el guion). El motor lo detecta solo: subtítulos y animaciones pasan a seguir la voz real.
`tts.py` avisa (⚠) si lo que entiende Whisper se parece poco al guion: suele ser una pronunciación mala; se corrige
añadiendo la palabra a `pronunciacion:` en el guion y regenerando solo ese segmento con `--solo s03`.

### Formato 2: pantalla en dos pisos (`formato: 2` en el guion)
Zonas (y, 1080×1920): serie + progreso por secciones 150–210 · título 225–300 · lema ~338 · **piso 1** (intuición, las escenas de siempre
desplazadas 100 px) 400–1100 · subtítulos ~1182 · **piso 2** (lo formal) 1252–1700. Sin tarjeta fija de usos/ojo: ahora es la **ficha**.
Guiones sin `formato: 2` se renderizan con el diseño anterior (`explainer_v1.py`). Piloto migrado: **Ep. 01**.

Claves nuevas del guion (ver `guion_ep01_mlp.yaml`):
| Clave | Dónde | Qué hace |
|---|---|---|
| `formato: 2` | raíz | activa los dos pisos |
| `lema` | raíz | frase fija bajo el título |
| `promesas: [a, b, c]` | raíz | texto del banner «EN ESTE VÍDEO» |
| `ficha: [[ETIQUETA, texto, color], …]` | raíz | ficha técnica (colores: verde, oro, rojo, azul, gris) |
| `pildora` / `pildora_oro: true` | segmento | resumen en el borde inferior del piso 1 |
| `piso2` | segmento | un tipo, o una lista de tipos que cambian en una palabra (`cue`) |

Tipos de `piso2`: `{tipo: banner}` · `{tipo: codigo, lineas: [..], ovalo: "texto exacto de la línea", ovalo_cue: palabra}` ·
`{tipo: formula, nombre: x, titulo: .., resalta: [partes], leyenda: [[símbolo, texto], ..]}` · `{tipo: lista, titulo: .., filas: [[izq, der], ..]}` ·
`{tipo: curva, nombre: x}` · `{tipo: ficha}`. El episodio aporta en `SCENES["_extras"]`: `code` (texto), `code_title`, `formulas` {nombre: Formula}
y `curves` {nombre: función(img, caja, segmento, t)}. El código del piso 2 muestra 10 líneas y se desplaza solo hasta las activas.
**Migrar un episodio** = añadir esas claves al YAML + el `_extras` en su `.py` (y quitar las etiquetas de promesas de su escena `hook`).

### Reglas de guion de los episodios largos (aplicarlas en cada episodio nuevo)
1. **Cada promesa del gancho tiene su escena de cierre.** Si el gancho enseña o anuncia algo («cómo pintó este cuadro»,
   «qué pasa sin color»), ese algo se explica y se muestra más adelante en el mismo episodio. Antes de renderizar,
   repasar la lista de promesas del segmento `s01` y señalar qué segmento cumple cada una.
2. **Presupuesto: 330–380 palabras** (≈ 2:30–2:50 a 2,4 palabras/s). Nunca más de 3:00 con la voz real.
3. **Una idea por segmento**, con su sección en pantalla. Orden: gancho → problema → intuición → fórmulas → código → demo real (con su curva de entrenamiento) → límites/cuándo usarla → siguiente arquitectura.
4. **Algo cambia en pantalla cada 2–3 s**, siempre disparado por una palabra de la narración (`cues`).
5. **Fórmulas término a término** y **código de ≤ 17 líneas**, resaltado por bloques cuando la voz los nombra.
6. **Honestidad visible**: simplificaciones y límites se dicen o se rotulan (p. ej. «aprende este cuadro de memoria»).
7. **El gancho se ve desde el fotograma 0** (sin fundido de entrada).
8. **El cierre enlaza con el siguiente episodio** a través de la limitación de la arquitectura actual.
9. **No ligar una animación a la última palabra de una frase**: aparecería justo cuando la escena termina. Usar una palabra anterior de la misma frase.
10. **(Técnica) No dibujar elementos con transparencia 0 sobre otros en la capa de la escena**: Pillow sustituye los píxeles en lugar de mezclarlos y «agujerea» lo que hay debajo. Saltar el dibujo si el alfa es ~0, o dibujar en una capa aparte y componerla.
11. **(Técnica) En el YAML, entrecomillar cualquier texto que contenga «: »** (p. ej. `saber: "inestables: dos redes…"`): si no, YAML lo interpreta como una clave y falla.
12. **(Técnica) Borrar `work/frames_<episodio>` tras codificar**: cada episodio ocupa ~0,6–1 GB en fotogramas y se regeneran con `chunk`.

**Para un episodio nuevo:** copiar el YAML y `ep01_mlp.py`, cambiar `arquitectura`, `titulo`, `tarjeta`, la narración
y las escenas. Presupuesto: ~330–380 palabras (2:30–2:50).
⚠ `tts.py` no se ha podido probar donde se escribió (sin GPU ni modelos): verificar la firma de `generate()` de la versión de Chatterbox instalada.

## 3. Versiones clásicas y «b» (formato panel)

Ver `run_all.sh`. Tabla resumida: Ep.1 espirales (`train.py`/`render_mlp.py`), 1b cuadro (`render_paint.py`),
2 dígitos / 2b cuadros, 3 texto / 3b pueblos, 4 atención (invertir palabra), 5 dígitos / 5b cuadros,
6 GAN 8 gaussianas, 7 club de kárate, 8 dígitos / 8b *La gran ola*.
⚠️ `render_mlp.py` es a la vez el Ep. 1 y el tema de los renders clásicos y **carga `work/run.pkl` al importarse**:
ejecutar `train.py` antes que cualquier `render_*.py`. Los Reels no tienen esta dependencia.

## 4. Notas honestas para la narración


- **1b**: la red **memoriza** el cuadro (PSNR medido sobre los mismos píxeles); el zoom ×8 **interpola**, no inventa detalle (lo dice el vídeo). Entrenado a 230×144.
- **2b**: reconoce **el cuadro, no al pintor** (una obra por artista). En B/N la red memoriza (casi 100 % en entrenamiento, 72 % en examen). El examen es la franja derecha de cada cuadro, nunca vista.
- **3b**: los nombres del reto están **elegidos a mano** entre muchas muestras (fijados en `select_pueblos.py`); ~95–97 % de lo generado no existe. Que alguien de cada comunidad revise que ningún nombre suene mal.
- **4**: Transformer mínimo (1 capa de atención, 1 cabeza, sin feed-forward).
- **5b**: con 2 números agrupa **por color**, no por estilo (Kandinsky queda en el centro y se excluye del paseo); reconstrucciones borrosas.
- **6**: quedan «puentes» entre modos; buen pie para hablar de *mode collapse*.
- **7**: el único fallo (nodo 8) tiene historia conocida en el estudio de Zachary; **verificarla** en el artículo original antes de narrarla.
- **8b**: los sueños son manchas de tinta, no espuma fina; con cadenas largas convergen al mismo patrón (por eso cada sueño se muestra a ~30 pasos desde ruido). Se necesitó más regularización para que aprendiera.
- En las versiones de dígitos (8): el error de la capa 2 no es comparable con el de la capa 1 (otra escala).

- **6 (Reel)**: la densidad de puntos es de **municipios, no de población**; Canarias, Ceuta y Melilla quedan fuera. El colapso que se ve es real (varios episodios; el peor ~iter 2.900).
- **7 (Reel)**: la red no ve coordenadas, pero el grafo de vecinos **sí se construye con ellas** (8 más cercanos). Decir siempre «Castilla y León» ya da un 28 %. Los fallos se concentran en fronteras del interior y enclaves (Treviño, Rincón de Ademuz).
- **Curvas suavizadas**: la de «aprende» en 2b (media de 5 puntos), las de pérdida en 3b/6/8b (media móvil/EMA). Las de examen se muestran tal cual.

## 5. Deuda técnica pendiente
1. Migrar `reel_paint.py` a `reel_common` (es el único Reel con su propio código de tipografías y CLI distinta).
2. Las versiones clásicas/«b» duplican cabeceras, gráficas y audio; los Reels ya no. Si se abandonan las clásicas, borrar `render_*.py`.
3. Imports con efectos secundarios en `train_cuadros.py`, `train_ae_cuadros.py`, `train_dbn_ola.py`, `train_pueblos.py`, `train_gan_es.py` (cargan datos al importarse).
4. Elecciones manuales: nombres del reto (`select_pueblos.py`), recorte del zoom (`BOX`), ejemplo de `A Arnoia`, enclaves señalados, semillas.
5. Rendimiento: Pillow en un hilo. En una máquina con varios núcleos se puede paralelizar `chunk` por rangos de fotogramas.

## 6. Licencias y fuentes
- **Cuadros** (`cuadros_pd/`): Van Gogh, Hokusai, Turner — dominio público; Munch y Kandinsky (†1944) — dominio público en España desde 1/1/2025. Imágenes tomadas de repositorios de transferencia de estilo en GitHub; para monetizar, usar copias de una fuente institucional (Wikimedia Commons, museos).
- **Municipios** (`data/`): paquete npm `spanish-cities-info` (ISC), datos verificados contra el INE, con coordenadas.
- **Fuentes** (`fonts/`): Archivo Black, Inter, Instrument Serif, IBM Plex Mono, Bricolage Grotesque, Space Grotesk, JetBrains Mono — SIL Open Font License 1.1 (licencias incluidas).
- **Club de kárate**: Zachary (1977) vía networkx. **Dígitos 8×8**: UCI vía scikit-learn (CC BY 4.0). **Audio**: sintetizado.

## 7. Próximos pasos sugeridos
- Ep. 4 en Reel: «¿Puede una red poner las tildes?» (Transformer + mapa de atención que mira al final de la palabra).
- `inventa.py` (Ep. 3b) y buscador de pueblo (Ep. 7) para responder a los comentarios.
- Voz en off (TTS local) + subtítulos grandes sincronizados por escena.
- Portar el entrenamiento a PyTorch en GPU para versiones a más resolución, manteniendo NumPy como «edición didáctica».
