"""Estilo de la serie, leído de config/estilo.yaml.

Las escenas importan de aquí los colores y tipografías; nunca usan hexadecimales.

    from lib.estilo import C, texto
    puntos.set_color(C.datos)
"""
from types import SimpleNamespace

from manim import ManimColor, Text, config as manim_config

from lib import config

_E = config.cargar("estilo")

FONDO = ManimColor(_E["fondo"])
# C.datos, C.activacion, C.gradiente, C.pesos, C.error, C.atencion,
# C.texto, C.texto_secundario, C.linea, C.panel
C = SimpleNamespace(**{k: ManimColor(v) for k, v in {**_E["colores"], **_E["neutros"]}.items()})

# CLASES[0], CLASES[1]…: colores de las clases de un dataset.
CLASES = [ManimColor(c) for c in _E["clases"]]

FUENTE = _E["tipografia"]["texto"]
FUENTE_CODIGO = _E["tipografia"]["codigo"]
TAM_TITULO = _E["tipografia"]["tamano_titulo"]
TAM_TEXTO = _E["tipografia"]["tamano_texto"]
TAM_NOTA = _E["tipografia"]["tamano_nota"]
MAX_PALABRAS = _E["texto_en_pantalla"]["max_palabras"]

manim_config.background_color = FONDO


def texto(contenido: str, tam: int = TAM_TEXTO, color: ManimColor = C.texto, **kw) -> Text:
    """Texto en pantalla con la tipografía de la serie. Falla si supera el máximo de palabras (§6.1)."""
    if len(contenido.split()) > MAX_PALABRAS:
        raise ValueError(f"Texto en pantalla con más de {MAX_PALABRAS} palabras: {contenido!r}")
    return Text(contenido, font=FUENTE, font_size=tam, color=color, **kw)
