"""Tarjetas "úsala cuando / evítala cuando": el bloque de honestidad técnica de cada episodio.

    tarjetas = TarjetaComparativa(
        usar=["Imágenes, audio, texto", "Muchos datos", "Patrones que se repiten"],
        evitar=["Tablas pequeñas", "Hay que explicar la decisión"])
    self.play(tarjetas.aparecer())
    self.play(tarjetas.rellenar("evitar"))
    self.play(tarjetas.rellenar("usar"))
"""
from manim import DOWN, LEFT, RIGHT, UP, AnimationGroup, FadeIn, RoundedRectangle, Text, VGroup

from lib.estilo import C, FUENTE, TAM_NOTA, TAM_TEXTO


class TarjetaComparativa(VGroup):
    def __init__(self, usar: list[str], evitar: list[str], ancho: float = 5.6, alto: float = 4.2,
                 titulo_usar: str = "Úsala cuando", titulo_evitar: str = "Evítala cuando", **kw):
        super().__init__(**kw)
        self.paneles, self.titulos, self.puntos = {}, {}, {}
        lados = [("evitar", titulo_evitar, evitar, C.error, -1), ("usar", titulo_usar, usar, C.activacion, 1)]
        for clave, titulo, items, color, signo in lados:
            panel = RoundedRectangle(width=ancho, height=alto, corner_radius=0.18, stroke_color=color,
                                     stroke_width=3, fill_color=C.panel, fill_opacity=1)
            panel.move_to([signo * (ancho / 2 + 0.25), 0, 0])
            cabecera = Text(titulo, font=FUENTE, font_size=TAM_TEXTO * 0.8, color=color)
            cabecera.next_to(panel.get_top(), DOWN, buff=0.35)
            lista = VGroup()
            for i, item in enumerate(items):
                fila = VGroup(Text("•", font=FUENTE, font_size=TAM_NOTA, color=color),
                              Text(item, font=FUENTE, font_size=TAM_NOTA, color=C.texto)).arrange(RIGHT, buff=0.22)
                fila.move_to(panel.get_left() + RIGHT * 0.45, aligned_edge=LEFT)
                fila.shift(UP * (panel.height / 2 - 1.5 - i * 0.7))
                lista.add(fila)
            self.paneles[clave], self.titulos[clave], self.puntos[clave] = panel, cabecera, lista
            self.add(panel, cabecera)

    def aparecer(self, **kw):
        return AnimationGroup(*[FadeIn(self.paneles[c], self.titulos[c], shift=UP * 0.15)
                                for c in self.paneles], lag_ratio=0.15, **kw)

    def rellenar(self, lado: str, **kw):
        """Los puntos de un lado aparecen uno a uno ('usar' o 'evitar')."""
        lista = self.puntos[lado]
        self.add(lista)
        return AnimationGroup(*[FadeIn(f, shift=RIGHT * 0.2) for f in lista], lag_ratio=0.35, **kw)
