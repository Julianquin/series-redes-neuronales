"""Red densa: columnas de neuronas con sus conexiones, y el flujo hacia delante y hacia atrás.

    red = RedDensa([2, 2, 2, 1])
    self.play(red.aparecer())
    self.play(red.animar_forward())    # activaciones, en verde azulado
    self.play(red.animar_backward())   # gradientes, en naranja
"""
import numpy as np
from manim import (
    Circle, Create, FadeIn, Line, LaggedStart, ShowPassingFlash, VGroup,
)

from lib.estilo import C


class RedDensa(VGroup):
    def __init__(self, capas: list[int], radio: float = 0.22, separacion_x: float = 1.9,
                 separacion_y: float = 0.75, **kw):
        super().__init__(**kw)
        self.capas = capas
        self.neuronas = VGroup()
        for i, n in enumerate(capas):
            columna = VGroup(*[
                Circle(radius=radio, color=C.activacion, stroke_width=3,
                       fill_color=C.panel, fill_opacity=1)
                .move_to([(i - (len(capas) - 1) / 2) * separacion_x,
                          ((n - 1) / 2 - j) * separacion_y, 0])
                for j in range(n)])
            self.neuronas.add(columna)

        self.conexiones = VGroup()
        for origen, destino in zip(self.neuronas, self.neuronas[1:]):
            for a in origen:
                for b in destino:
                    self.conexiones.add(Line(a.get_right(), b.get_left(), stroke_width=1.6,
                                             color=C.linea, stroke_opacity=0.7))
        self.add(self.conexiones, self.neuronas)

    def aparecer(self, **kw):
        return LaggedStart(*[FadeIn(c) for c in self.neuronas],
                           Create(self.conexiones), lag_ratio=0.2, **kw)

    def _destellos(self, color, inverso: bool, **kw):
        grupos = []
        for i in range(len(self.capas) - 1):
            tramo = [l for l in self.conexiones
                     if self._capa_de(l) == (len(self.capas) - 2 - i if inverso else i)]
            grupos.append(VGroup(*tramo))
        return LaggedStart(*[
            ShowPassingFlash(g.copy().set_color(color).set_stroke(width=4),
                             time_width=0.6, reverse_rate_function=inverso)
            for g in grupos], lag_ratio=0.45, **kw)

    def _capa_de(self, linea: Line) -> int:
        """A qué tramo entre columnas pertenece una conexión (por su coordenada x)."""
        x = linea.get_center()[0]
        centros = [c.get_center()[0] for c in self.neuronas]
        return int(np.argmin([abs(x - (a + b) / 2) for a, b in zip(centros, centros[1:])]))

    def animar_forward(self, **kw):
        """Activaciones avanzando capa a capa."""
        return self._destellos(C.activacion, inverso=False, **kw)

    def animar_backward(self, **kw):
        """Gradientes volviendo hacia atrás (§6.1: naranja)."""
        return self._destellos(C.gradiente, inverso=True, **kw)
