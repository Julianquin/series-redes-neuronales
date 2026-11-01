"""El error como paisaje: superficie 3D y una bola que baja por el gradiente.

Para escenas EscenaNarrada3D. El descenso es un descenso de gradiente de verdad sobre la
función que se dibuja, no una trayectoria inventada.

    paisaje = PaisajePerdida()
    self.add(paisaje, paisaje.bola)
    self.play(paisaje.descender(pasos=12), run_time=3)
"""
import numpy as np
from manim import Dot3D, Succession, Surface, VGroup

from lib.estilo import C


def _valle(x: float, y: float) -> float:
    """Paisaje con un valle claro y un poco de relieve alrededor."""
    return 0.45 * (x ** 2 + 1.6 * y ** 2) - 0.6 * np.exp(-((x - 1.2) ** 2 + (y + 1.0) ** 2)) \
        + 0.25 * np.sin(1.7 * x) * np.cos(1.5 * y)


class PaisajePerdida(VGroup):
    def __init__(self, funcion=_valle, rango=(-2.4, 2.4), inicio=(-2.0, 1.7), altura: float = 0.55,
                 resolucion: int = 28, **kw):
        super().__init__(**kw)
        self.f = funcion
        self.altura = altura
        # Damero suave: el rojo es del error (§6.1), pero a plena intensidad tapa la bola.
        self.superficie = Surface(
            lambda u, v: np.array([u, v, altura * self.f(u, v)]),
            u_range=rango, v_range=rango, resolution=(resolucion, resolucion),
            fill_opacity=0.72, checkerboard_colors=[C.error, C.panel], stroke_width=0.8,
            stroke_color=C.linea,
        ).set_fill(opacity=0.72)
        self.add(self.superficie)
        self.posicion = np.array(inicio, dtype=float)
        self.bola = Dot3D(self._punto(self.posicion), radius=0.13, color=C.datos)

    def _punto(self, xy: np.ndarray) -> np.ndarray:
        """Centro de la bola: sobre la superficie, levantada su propio radio."""
        return np.array([xy[0], xy[1], self.altura * self.f(xy[0], xy[1]) + 0.13])

    def gradiente(self, xy: np.ndarray, h: float = 1e-3) -> np.ndarray:
        x, y = xy
        return np.array([(self.f(x + h, y) - self.f(x - h, y)) / (2 * h),
                         (self.f(x, y + h) - self.f(x, y - h)) / (2 * h)])

    def descender(self, pasos: int = 12, tasa: float = 0.18, run_time_paso: float = 0.25):
        """Descenso de gradiente real: cada paso es una animación corta, así se ve el avance."""
        animaciones = []
        for _ in range(pasos):
            self.posicion = self.posicion - tasa * self.gradiente(self.posicion)
            animaciones.append(self.bola.animate(run_time=run_time_paso).move_to(self._punto(self.posicion)))
        return Succession(*animaciones)
