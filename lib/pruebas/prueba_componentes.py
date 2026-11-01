"""Escenas de prueba de los componentes (CLAUDE.md: cada componente nuevo lleva la suya).

    .venv/bin/manim -ql -s lib/pruebas/prueba_componentes.py PruebaNubePuntos
"""
from manim import Create, FadeIn, FadeOut, Scene, Transform

from lib import estilo  # noqa: F401  (fondo de la serie)
from lib.componentes.neurona import Neurona
from lib.componentes.nube_puntos import FronteraDecision, NubePuntos


class PruebaNubePuntos(Scene):
    def construct(self):
        nube = NubePuntos.lunas()
        recta = FronteraDecision.recta(nube)
        self.play(FadeIn(nube), Create(recta))
        self.play(Transform(recta, FronteraDecision.curva(nube)))


class PruebaNeurona(Scene):
    def construct(self):
        n = Neurona()
        self.play(n.aparecer())
        self.play(n.mostrar_pesos())
        self.play(n.mostrar_suma())
        self.play(n.mostrar_activacion())


class PruebaEjesYRejilla(Scene):
    """Gancho del ep00: ejes con unidades, y la capa deformando el espacio."""

    def construct(self):
        from manim import Write
        from lib.componentes.ejes_etiquetados import EjesEtiquetados
        from lib.componentes.rejilla_deformable import RejillaDeformable

        nube = NubePuntos.lunas(n=120, escala=1.6)
        ejes = EjesEtiquetados(nube, "Temperatura (°C)", "Humedad (%)", (18, 32), (40, 90))
        self.play(Create(ejes), FadeIn(nube), Write(ejes.nota))
        self.wait(0.3)
        self.play(FadeOut(ejes, ejes.nota))

        rejilla = RejillaDeformable(nube, extension=5)
        self.add(rejilla)
        self.play(rejilla.animar(W=[[1.3, -0.9], [0.7, 1.2]]), run_time=1.5)   # pesos
        self.play(rejilla.animar(b=[0.6, -0.4]), run_time=1.0)                 # sesgo
        self.play(rejilla.animar(relu=True), run_time=1.5)                     # activación


class PruebaRedDensa(Scene):
    def construct(self):
        from lib.componentes.red_densa import RedDensa

        red = RedDensa([2, 4, 4, 1])
        self.play(red.aparecer(), run_time=1.5)
        self.play(red.animar_forward(), run_time=1.5)
        self.play(red.animar_backward(), run_time=1.5)


class PruebaTarjetas(Scene):
    def construct(self):
        from lib.componentes.tarjeta_comparativa import TarjetaComparativa

        t = TarjetaComparativa(
            usar=["Imágenes, audio, texto", "Muchos datos", "Patrones que se repiten"],
            evitar=["Tablas pequeñas", "Hay que explicar por qué", "Entrenar cuesta caro"])
        self.play(t.aparecer())
        self.play(t.rellenar("evitar"))
        self.play(t.rellenar("usar"))


class PruebaMatriz(Scene):
    def construct(self):
        from lib.componentes.matriz import Matriz

        m = Matriz.imagen(10, 10, lado=0.45)
        self.play(m.aparecer(), run_time=1.2)
        self.play(m.resaltar([(4, j) for j in range(10)]), run_time=0.8)
        self.play(m.aplanar(), run_time=2)


class PruebaFlechaGradiente(Scene):
    def construct(self):
        from manim import GrowArrow, ORIGIN

        from lib.componentes.flecha_gradiente import FlechaGradiente

        g = FlechaGradiente(ORIGIN, (2.0, 1.0), etiqueta=r"\nabla L")
        self.play(GrowArrow(g.flecha), FadeIn(g.etiqueta))
        for v in ((1.0, 0.5), (0.35, 0.18), (0.08, 0.04)):
            self.play(g.actualizar(v), run_time=0.7)
