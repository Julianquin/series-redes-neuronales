"""Ejes con nombre y unidades para una nube de puntos, y la nota de "datos de ejemplo".

Los datos sintéticos siempre se marcan como tales (§17 del plan: honestidad con la audiencia).

    ejes = EjesEtiquetados(nube, "Temperatura (°C)", "Humedad (%)",
                           rango_x=(18, 32), rango_y=(40, 90))
    self.play(Create(ejes), FadeIn(ejes.nota))
"""
import numpy as np
from manim import DL, DOWN, LEFT, ORIGIN, Line, Text, VGroup

from lib.estilo import C, FUENTE, TAM_NOTA


class EjesEtiquetados(VGroup):
    """Dos ejes en L alrededor de una nube, con etiquetas y marcas de valor en los extremos.

    Los rangos son los de los datos "reales" (grados, por ciento); la nube vive en coordenadas
    normalizadas, así que solo se etiquetan los extremos, sin cuadrícula que distraiga.
    """

    def __init__(self, nube, nombre_x: str, nombre_y: str, rango_x: tuple[float, float],
                 rango_y: tuple[float, float], margen: float = 0.6, nota: str = "datos de ejemplo", **kw):
        super().__init__(**kw)
        puntos = np.array([p.get_center() for p in nube])
        x0, y0 = puntos[:, 0].min() - margen, puntos[:, 1].min() - margen
        x1, y1 = puntos[:, 0].max() + margen, puntos[:, 1].max() + margen

        self.eje_x = Line([x0, y0, 0], [x1, y0, 0], color=C.linea, stroke_width=3)
        self.eje_y = Line([x0, y0, 0], [x0, y1, 0], color=C.linea, stroke_width=3)
        etiqueta = lambda t: Text(t, font=FUENTE, font_size=TAM_NOTA * 0.8, color=C.texto_secundario)
        self.nombre_x = etiqueta(nombre_x).next_to(self.eje_x, DOWN, buff=0.35)
        self.nombre_y = etiqueta(nombre_y).rotate(np.pi / 2).next_to(self.eje_y, LEFT, buff=0.35)
        self.valores = VGroup(
            etiqueta(f"{rango_x[0]:g}").next_to([x0, y0, 0], DOWN, buff=0.15),
            etiqueta(f"{rango_x[1]:g}").next_to([x1, y0, 0], DOWN, buff=0.15),
            etiqueta(f"{rango_y[0]:g}").next_to([x0, y0, 0], LEFT, buff=0.15),
            etiqueta(f"{rango_y[1]:g}").next_to([x0, y1, 0], LEFT, buff=0.15),
        )
        self.add(self.eje_x, self.eje_y, self.nombre_x, self.nombre_y, self.valores)

        # La nota va fuera del grupo: se muestra y se quita por separado.
        self.nota = etiqueta(nota).scale(0.85).to_corner(DL, buff=0.35) if nota else None

    def centrar(self, *mobs) -> "EjesEtiquetados":
        """Centra en pantalla los ejes junto con lo que contengan (la nube crece solo a un lado,
        así que sin esto la composición queda desplazada)."""
        VGroup(self, *mobs).move_to(ORIGIN)
        return self
