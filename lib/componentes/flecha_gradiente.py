"""Flecha de gradiente: naranja, con la longitud proporcional a su magnitud.

Si el gradiente es pequeño, la flecha se ve corta y translúcida; así se entiende de un vistazo
el desvanecimiento del gradiente (útil también en el episodio 02, con las RNN).

    g = FlechaGradiente(origen, (0.8, -0.4))
    self.play(GrowArrow(g))
    self.play(g.actualizar((0.1, -0.05)))    # el gradiente se apaga
"""
import numpy as np
from manim import Arrow, Transform, VGroup

from lib.estilo import C, TAM_NOTA


class FlechaGradiente(VGroup):
    def __init__(self, origen, vector, escala: float = 1.2, etiqueta: str | None = None, **kw):
        super().__init__(**kw)
        self.origen = np.array([*origen, 0][:3], dtype=float)
        self.escala = escala
        self.flecha = self._flecha(vector)
        self.add(self.flecha)
        if etiqueta:
            from manim import MathTex

            self.etiqueta = MathTex(etiqueta, color=C.gradiente, font_size=TAM_NOTA * 1.2)
            # Junto a la mitad de la flecha y perpendicular a ella: así no se despega al acortarse.
            d = self.flecha.get_unit_vector()
            self.etiqueta.move_to(self.flecha.point_from_proportion(0.5) + np.array([-d[1], d[0], 0]) * 0.35)
            self.add(self.etiqueta)

    def _flecha(self, vector) -> Arrow:
        v = np.array([*vector, 0][:3], dtype=float)
        magnitud = float(np.linalg.norm(v))
        # La opacidad baja con la magnitud: un gradiente pequeño casi no se ve.
        opacidad = float(np.clip(0.25 + magnitud, 0.25, 1.0))
        return Arrow(self.origen, self.origen + v * self.escala, buff=0,
                     color=C.gradiente, stroke_width=6, max_tip_length_to_length_ratio=0.28,
                     stroke_opacity=opacidad, fill_opacity=opacidad)

    def actualizar(self, vector, **kw):
        """Animación hacia un gradiente nuevo (misma base, otra longitud y opacidad)."""
        return Transform(self.flecha, self._flecha(vector), **kw)
