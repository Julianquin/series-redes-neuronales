"""Prueba de la Fase 0: Manim + LaTeX con texto en español."""
from manim import DOWN, UP, FadeIn, MathTex, Scene, Tex, Write


class PruebaLatex(Scene):
    def construct(self):
        titulo = Tex(r"Función de pérdida y actualización").to_edge(UP)
        ecuacion = MathTex(r"w \leftarrow w - \eta \, \frac{\partial L}{\partial w}")
        atencion = MathTex(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
        atencion.next_to(ecuacion, DOWN, buff=0.8)
        self.play(FadeIn(titulo))
        self.play(Write(ecuacion))
        self.play(Write(atencion))
        self.wait(0.5)
