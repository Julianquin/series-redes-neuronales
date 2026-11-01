"""Escenas del episodio de prueba (ep99): una por segmento de guion.yaml."""
from manim import DOWN, UP, Create, FadeIn, Rotate, Transform, Write

from lib.componentes.neurona import Neurona
from lib.componentes.nube_puntos import FronteraDecision, NubePuntos
from lib.escena_narrada import EscenaNarrada
from lib.estilo import C, TAM_TITULO, texto


class PruebaGancho(EscenaNarrada):
    """s01 · «Mira estos puntos. Ninguna línea recta puede separarlos. Y sin embargo, una red
    neuronal encontrará la frontera ella sola.»"""
    segmento = "s01"

    def construct(self):
        nube = NubePuntos.lunas()
        self.play(FadeIn(nube, lag_ratio=0.01), run_time=1.2)

        self.hasta_marca("aparece_recta")
        recta = FronteraDecision.recta(nube, pendiente=-0.4)
        self.play(Create(recta), run_time=0.8)
        # La recta prueba inclinaciones y ninguna separa las dos clases.
        for angulo in (0.5, -0.8, 0.3):
            self.play(Rotate(recta, angulo), run_time=0.7)

        self.hasta_marca("frontera_empieza")
        self.play(Transform(recta, FronteraDecision.curva(nube)), run_time=self.tiempo_final(maximo=1.6))
        self.completar()


class PruebaNeurona(EscenaNarrada):
    """s02 · «Una neurona multiplica cada entrada por un peso, suma el resultado y lo pasa por una
    función de activación llamada ReLU.»"""
    segmento = "s02"

    def construct(self):
        neurona = Neurona()
        self.play(neurona.aparecer(), run_time=2.0)

        self.hasta_marca("mostrar_pesos")
        self.play(neurona.mostrar_pesos(), run_time=self.tiempo_hasta("mostrar_suma", maximo=0.8))

        self.hasta_marca("mostrar_suma")
        self.play(neurona.mostrar_suma(), run_time=0.8)

        self.hasta_marca("mostrar_relu")
        self.play(neurona.mostrar_activacion(), run_time=self.tiempo_final(maximo=1.0))
        self.completar()


class PruebaPuente(EscenaNarrada):
    """s03 · «En el próximo vídeo veremos por qué esto falla con imágenes.»"""
    segmento = "s03"

    def construct(self):
        pregunta = texto("¿Y con imágenes?", tam=TAM_TITULO)
        nota = texto("Próximo episodio: CNN", color=C.texto_secundario).next_to(pregunta, DOWN, buff=0.5)
        self.play(Write(pregunta), run_time=1.2)
        self.play(FadeIn(nota, shift=UP * 0.2), run_time=0.6)
        self.completar()
