"""Rejilla de valores: una matriz, una imagen en píxeles o un vector.

Se usa para resaltar filas, columnas y celdas, y para aplanar una imagen en una fila larguísima
de números (el puente del episodio 00 a las CNN).

    m = Matriz.imagen(12, 12)                 # "foto" en escala de grises
    self.play(m.aparecer())
    self.play(m.aplanar(), run_time=2)        # se pierde la estructura espacial
"""
import numpy as np
from manim import (
    AnimationGroup, FadeIn, LaggedStart, Square, Text, Transform, VGroup,
)

from lib.estilo import C, FUENTE, TAM_NOTA


class Matriz(VGroup):
    def __init__(self, valores: np.ndarray, lado: float = 0.52, mostrar_valores: bool = True,
                 color=None, **kw):
        super().__init__(**kw)
        self.valores = np.asarray(valores, dtype=float)
        self.lado = lado
        filas, columnas = self.valores.shape
        self.celdas = VGroup()
        for i in range(filas):
            for j in range(columnas):
                v = self.valores[i, j]
                celda = Square(side_length=lado, stroke_width=1.5, stroke_color=C.linea,
                               fill_color=color or C.datos, fill_opacity=float(np.clip(v, 0, 1)) * 0.85)
                celda.move_to([(j - (columnas - 1) / 2) * lado, ((filas - 1) / 2 - i) * lado, 0])
                if mostrar_valores:
                    etiqueta = Text(f"{v:.1f}", font=FUENTE, font_size=TAM_NOTA * lado * 1.3, color=C.texto)
                    etiqueta.move_to(celda)
                    celda = VGroup(celda, etiqueta)
                self.celdas.add(celda)
        self.filas, self.columnas = filas, columnas
        self.add(self.celdas)

    @classmethod
    def imagen(cls, filas: int, columnas: int, semilla: int = 0, **kw) -> "Matriz":
        """Rejilla tipo foto: valores suaves, sin números encima."""
        rng = np.random.default_rng(semilla)
        y, x = np.mgrid[0:filas, 0:columnas] / max(filas, columnas)
        base = 0.5 + 0.4 * np.sin(4 * x + 1.2) * np.cos(3.5 * y)
        return cls(np.clip(base + rng.normal(0, 0.06, base.shape), 0, 1), mostrar_valores=False, **kw)

    def celda(self, i: int, j: int) -> VGroup:
        return self.celdas[i * self.columnas + j]

    def aparecer(self, **kw):
        return LaggedStart(*[FadeIn(c) for c in self.celdas], lag_ratio=0.01, **kw)

    def resaltar(self, celdas: list[tuple[int, int]], color=None, **kw):
        """Resalta celdas concretas (una fila, una columna o una ventana)."""
        color = color or C.atencion
        objetivos = [self.celda(i, j) for i, j in celdas]
        return AnimationGroup(*[c.animate.set_stroke(color, width=4) for c in objetivos], **kw)

    def aplanar(self, ancho_maximo: float = 12.0, **kw):
        """Convierte la rejilla en una sola fila: la estructura espacial se pierde."""
        n = len(self.celdas)
        lado = min(self.lado, ancho_maximo / n)
        destino = VGroup()
        for k, celda in enumerate(self.celdas):
            copia = celda.copy().scale(lado / self.lado)
            copia.move_to([(k - (n - 1) / 2) * lado, 0, 0])
            destino.add(copia)
        return Transform(self.celdas, destino, **kw)
