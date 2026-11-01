"""Nube de puntos de un dataset 2D y frontera de decisión.

    nube = NubePuntos.lunas()
    recta = FronteraDecision.recta(nube)
    curva = FronteraDecision.curva(nube)
    self.play(Transform(recta, curva))     # la recta se dobla hasta separar las clases
"""
import numpy as np
from manim import Dot, ParametricFunction, VGroup

from lib.estilo import CLASES, C


class NubePuntos(VGroup):
    """Puntos coloreados por clase. Guarda la transformación datos → escena para dibujar
    fronteras en las mismas coordenadas."""

    def __init__(self, xy: np.ndarray, clases: np.ndarray, centro=(0.0, 0.0), escala: float = 1.0,
                 radio: float = 0.06, **kw):
        super().__init__(**kw)
        self.centro = np.asarray(centro, dtype=float)
        self.escala = escala
        self.xy, self.clases = xy, clases
        for (x, y), c in zip(xy, clases):
            self.add(Dot(self.a_escena(x, y), radius=radio, color=CLASES[int(c)]))

    @property
    def xy_escena(self) -> np.ndarray:
        """Coordenadas en pantalla (self.xy son las de los datos: grados, por ciento…)."""
        return np.array([p.get_center()[:2] for p in self])

    def a_escena(self, x: float, y: float) -> np.ndarray:
        return np.array([(x - self.centro[0]) * self.escala, (y - self.centro[1]) * self.escala, 0.0])

    @classmethod
    def lunas(cls, n: int = 160, ruido: float = 0.08, semilla: int = 0, escala: float = 2.2, **kw) -> "NubePuntos":
        """Dos medias lunas entrelazadas (equivalente a sklearn.datasets.make_moons)."""
        rng = np.random.default_rng(semilla)
        m = n // 2
        t_a, t_b = np.linspace(0, np.pi, m), np.linspace(0, np.pi, n - m)
        a = np.column_stack([np.cos(t_a), np.sin(t_a)])
        b = np.column_stack([1 - np.cos(t_b), 0.5 - np.sin(t_b)])
        xy = np.vstack([a, b]) + rng.normal(0, ruido, (n, 2))
        clases = np.array([0] * m + [1] * (n - m))
        return cls(xy, clases, centro=(0.5, 0.25), escala=escala, **kw)


class FronteraDecision:
    """Fronteras en coordenadas de datos de una NubePuntos. Todas se muestrean igual para que
    Transform entre ellas sea suave."""

    RANGO_X = (-1.4, 2.4)

    @classmethod
    def _curva(cls, nube: NubePuntos, f, color=None) -> ParametricFunction:
        return ParametricFunction(lambda t: nube.a_escena(t, f(t)), t_range=(*cls.RANGO_X, 0.02),
                                  color=color or C.texto, stroke_width=5)

    @classmethod
    def recta(cls, nube: NubePuntos, pendiente: float = 0.0, corte: float = 0.25, **kw) -> ParametricFunction:
        return cls._curva(nube, lambda x: pendiente * (x - 0.5) + corte, **kw)

    @classmethod
    def curva(cls, nube: NubePuntos, **kw) -> ParametricFunction:
        """Frontera que separa las lunas: pasa entre ambas en todo el rango."""
        return cls._curva(nube, lambda x: 0.25 + 0.55 * np.sin(np.pi * (x + 0.5)), **kw)
