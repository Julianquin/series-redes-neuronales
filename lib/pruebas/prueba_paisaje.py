"""Prueba del paisaje de la pérdida (escena 3D)."""
from manim import DEGREES, Create, ThreeDScene

from lib import estilo  # noqa: F401
from lib.componentes.paisaje_perdida import PaisajePerdida


class PruebaPaisaje(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=58 * DEGREES, theta=-50 * DEGREES, zoom=0.65)
        paisaje = PaisajePerdida()
        self.play(Create(paisaje.superficie), run_time=1.5)
        self.add(paisaje.bola)
        self.play(paisaje.descender(pasos=14), run_time=3)
