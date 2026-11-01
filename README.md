# Serie de vídeos: arquitecturas de redes neuronales

Pipeline local que convierte `episodios/<ep>/guion.yaml` en un vídeo narrado y animado con Manim.
El plan completo está en `plan_serie_redes_neuronales.md`; las reglas de trabajo, en `CLAUDE.md`.

## Entorno

```bash
./env/setup_wsl.sh                      # apt + .venv (Python 3.13, torch cu128) + verificación
.venv/bin/python env/verificar_entorno.py
```

## Uso

```bash
.venv/bin/python pipeline/produce.py --ep 99              # hasta el primer punto de control
.venv/bin/python pipeline/produce.py --ep 99 --desde e2   # continuar tras revisar el guion
.venv/bin/python pipeline/produce.py --ep 99 --solo e1    # solo validar el guion
```

## Estado

| Etapa | Estado |
|---|---|
| e1 Validar guion | ✅ |
| e2 TTS y QA de voz | ✅ motor **Qwen3-TTS** (en `.venv-qwen`); Chatterbox sigue disponible en `config/voz.yaml` |
| e3 Escenas (Claude Code) | ✅ comprobación de clases; escenas del ep99 |
| e4 Render | ✅ paralelo, con caché y control de desfase |
| e5 Hoja de contactos | ✅ `build/contactos/<calidad>/hoja_N.png` |
| e6 Ensamblado | ✅ voz + música con ducking, −14 LUFS, NVENC |
| e7 Subtítulos y capítulos | ✅ `.srt` desde el guion, capítulos por bloque |
| e8 QA final | ✅ `salida/epXX/informe_qa.md` |
| e9 Miniatura y metadatos | ⏳ |

Prueba completa sin puntos de control: `.venv/bin/python pipeline/produce.py --ep 99 --hasta e8 --sin-pausas --borrador`

## Mejoras pendientes

- **Efectos de sonido puntuales.** Un campo `sfx` en las marcas del guion y una mezcla en e6
  (los tiempos exactos ya los da e2). Decidido: se valora **después** del piloto, para añadirlos
  solo donde aporten. Dos o tres por episodio como máximo; cada archivo con su licencia.
- **e9: miniatura y metadatos** (ComfyUI + FLUX.1 schnell), sin instalar todavía.
- **Demo real del episodio 00** (MLP en make_moons y comparación con gradient boosting).
- **Material visual generado** (FLUX.2 klein para miniaturas, Wan 2.2 para B-roll): en evaluación
  (§20 del plan), nada de esto entra en un episodio sin decidirlo antes. La prueba de Wan está en
  `notebooks/04_probar_wan.ipynb`, que mide el coste real antes de comprometerse.
