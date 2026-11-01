"""Una neurona construida paso a paso: entradas, pesos, suma y activación ReLU.

    n = Neurona(entradas=[0.5, -1.2, 0.8], pesos=[0.9, 0.3, -0.7])
    self.play(n.aparecer())
    self.play(n.mostrar_pesos())
    self.play(n.mostrar_suma())
    self.play(n.mostrar_activacion())
"""
import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, AnimationGroup, Arrow, Axes, Circle, Create, DecimalNumber, FadeIn, GrowArrow,
    LaggedStart, MathTex, VGroup, Write,
)

from lib.estilo import C, TAM_NOTA


class Neurona(VGroup):
    def __init__(self, entradas=(0.5, -1.2, 0.8), pesos=(0.9, 0.3, -0.7), sesgo: float = 0.4,
                 radio: float = 0.75, **kw):
        super().__init__(**kw)
        self.cuerpo = Circle(radius=radio, color=C.activacion, stroke_width=5)

        separacion = 1.4
        self.valores, self.flechas, self.pesos = VGroup(), VGroup(), VGroup()
        for i, (x, w) in enumerate(zip(entradas, pesos)):
            y = (len(entradas) - 1) / 2 * separacion - i * separacion
            valor = DecimalNumber(x, num_decimal_places=1, include_sign=x < 0, color=C.datos,
                                  font_size=TAM_NOTA * 1.4).move_to([-4.2, y, 0])
            destino = self.cuerpo.point_at_angle(np.pi - ((len(entradas) - 1) / 2 - i) * 0.5)
            flecha = Arrow(valor.get_right() + RIGHT * 0.1, destino, buff=0.05, color=C.linea, stroke_width=4)
            peso = MathTex(f"w_{i + 1}={w:g}", color=C.pesos, font_size=TAM_NOTA * 1.4)
            # Etiqueta en perpendicular a la flecha, sin tocarla: encima salvo la última, debajo.
            direccion = flecha.get_unit_vector()
            normal = np.array([-direccion[1], direccion[0], 0.0])
            if i == len(entradas) - 1:
                normal = -normal
            distancia = 0.12 + max(peso.width * abs(normal[0]), peso.height * abs(normal[1])) / 2 + 0.1
            peso.move_to(flecha.point_from_proportion(0.4) + normal * distancia)
            self.valores.add(valor)
            self.flechas.add(flecha)
            self.pesos.add(peso)

        self.suma = MathTex(r"\Sigma", color=C.texto, font_size=72).move_to(self.cuerpo)
        # El sesgo entra como un término aparte, debajo del cuerpo.
        self.sesgo = MathTex(rf"+\,b={sesgo:g}", color=C.pesos, font_size=TAM_NOTA * 1.4)
        self.sesgo.next_to(self.cuerpo, DOWN, buff=0.3)
        self.salida = Arrow(self.cuerpo.get_right(), self.cuerpo.get_right() + RIGHT * 1.3, buff=0.05,
                            color=C.activacion, stroke_width=4)

        ejes = Axes(x_range=(-2, 2, 1), y_range=(0, 2, 1), x_length=1.8, y_length=1.2,
                    axis_config={"color": C.linea, "include_ticks": False, "stroke_width": 2})
        grafica = ejes.plot(lambda t: max(0.0, t), x_range=(-2, 2), color=C.activacion, stroke_width=5)
        nombre = MathTex(r"\mathrm{ReLU}", color=C.activacion, font_size=TAM_NOTA * 1.3).next_to(ejes, DOWN, buff=0.15)
        self.relu = VGroup(ejes, grafica, nombre).next_to(self.salida, RIGHT, buff=0.25)

        # Centrada según la composición completa, aunque pesos, suma y activación aún no estén en pantalla.
        VGroup(self.valores, self.flechas, self.pesos, self.cuerpo, self.suma, self.sesgo,
               self.salida, self.relu).move_to(ORIGIN)
        # Solo lo básico forma parte del grupo; pesos, suma y activación entran con sus animaciones.
        self.add(self.valores, self.flechas, self.cuerpo)

    def aparecer(self, **kw):
        """Todo de una vez: cuerpo, entradas y flechas."""
        return LaggedStart(Create(self.cuerpo), FadeIn(self.valores, shift=RIGHT * 0.3),
                           *[GrowArrow(f) for f in self.flechas], lag_ratio=0.25, **kw)

    def aparecer_cuerpo(self, **kw):
        return Create(self.cuerpo, **kw)

    def entrar_valores(self, **kw):
        """Las entradas llegan y las flechas crecen (para sincronizar con "recibe unos números")."""
        return LaggedStart(FadeIn(self.valores, shift=RIGHT * 0.3),
                           *[GrowArrow(f) for f in self.flechas], lag_ratio=0.2, **kw)

    def destacar_relu(self, **kw):
        """Resalta que la ReLU aplasta lo negativo: parpadea su curva."""
        from manim import Indicate

        return Indicate(self.relu[1], color=C.activacion, scale_factor=1.15, **kw)

    def mostrar_pesos(self):
        return AnimationGroup(*[FadeIn(p, shift=DOWN * 0.15) for p in self.pesos],
                              *[f.animate.set_color(C.pesos) for f in self.flechas], lag_ratio=0.2)

    def mostrar_suma(self):
        return Write(self.suma)

    def mostrar_sesgo(self):
        return FadeIn(self.sesgo, shift=UP * 0.15)

    def mostrar_activacion(self):
        return AnimationGroup(GrowArrow(self.salida), Create(self.relu), lag_ratio=0.4)
