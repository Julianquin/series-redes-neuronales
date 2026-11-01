"""Escenas del episodio 00 (piloto): qué aprende realmente una red neuronal.

Las deformaciones y la frontera salen de los pesos entrenados en demo/entrenar_capas.py:
lo que se ve es la transformación que la red aprendió, no una inventada.

Cada marca del guion tiene aquí su momento visual: el objetivo es que nunca pasen más de
unos segundos sin que cambie algo en pantalla.
"""
import json
from pathlib import Path

import numpy as np
from manim import (
    DEGREES, DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow3D, ChangeDecimalToValue, Circle, Create,
    FadeIn, FadeOut, Indicate, Integer, MathTex, ParametricFunction, Polygon, Text, Transform,
    VGroup, Write,
)

from lib.componentes.ejes_etiquetados import EjesEtiquetados
from lib.componentes.matriz import Matriz
from lib.componentes.neurona import Neurona
from lib.componentes.nube_puntos import FronteraDecision, NubePuntos
from lib.componentes.paisaje_perdida import PaisajePerdida
from lib.componentes.red_densa import RedDensa
from lib.componentes.rejilla_deformable import RejillaDeformable
from lib.componentes.tarjeta_comparativa import TarjetaComparativa
from lib.escena_narrada import EscenaNarrada, EscenaNarrada3D
from lib.estilo import C, CLASES, FUENTE, TAM_NOTA, TAM_TEXTO, TAM_TITULO, texto

PESOS = json.loads((Path(__file__).parent / "demo" / "pesos_invernadero.json").read_text(encoding="utf-8"))
UNA_CAPA, DOS_CAPAS = PESOS["1_capa"], PESOS["2_capas"]
ESCALA_NUBE = 1.6


# --- datos y red -----------------------------------------------------------------------------

def datos(n: int = 200, **kw) -> NubePuntos:
    """La misma nube con la que se entrenó (ver demo/entrenar_capas.py)."""
    return NubePuntos.lunas(n=n, escala=ESCALA_NUBE, **kw)


def red(xy: np.ndarray, pesos: dict, capas: int) -> np.ndarray:
    h = xy
    for i in range(1, capas + 1):
        h = np.maximum(h @ np.array(pesos[f"W{i}"]).T + np.array(pesos[f"b{i}"]), 0)
    return h


def salida(xy: np.ndarray, pesos: dict, capas: int) -> np.ndarray:
    return red(xy, pesos, capas) @ np.array(pesos["v"]) + np.array(pesos["c"])


def frontera_en_datos(pesos: dict, capas: int, n: int = 220) -> np.ndarray:
    """La frontera de decisión en coordenadas de pantalla, buscada columna a columna."""
    puntos = []
    for x in np.linspace(-3.2, 3.2, n):
        ys = np.linspace(-2.6, 2.6, 400)
        z = salida(np.column_stack([np.full_like(ys, x), ys]), pesos, capas)
        for k in np.flatnonzero(np.sign(z[:-1]) != np.sign(z[1:])):
            puntos.append([x, (ys[k] + ys[k + 1]) / 2])
    return np.array(puntos)


def curva_frontera(pesos: dict = DOS_CAPAS, capas: int = 2) -> ParametricFunction:
    """La frontera aprendida: quebrada, porque la red usa ReLU."""
    puntos = frontera_en_datos(pesos, capas)
    puntos = puntos[np.argsort(puntos[:, 0])]

    def punto(t: float) -> np.ndarray:
        return np.array([*puntos[int(np.clip(t, 0, len(puntos) - 1))], 0.0])

    return ParametricFunction(punto, t_range=(0, len(puntos) - 1, 1), color=C.texto, stroke_width=5)


# --- piezas de apoyo -------------------------------------------------------------------------

def contador(titulo: str, valor: str, color) -> VGroup:
    grupo = VGroup(
        Text(titulo, font=FUENTE, font_size=TAM_NOTA, color=C.texto_secundario),
        Text(valor, font=FUENTE, font_size=TAM_TEXTO, color=color),
    ).arrange(DOWN, buff=0.12)
    return grupo.to_corner(UP + RIGHT, buff=0.6)


def etiqueta_error(valor: float) -> VGroup:
    return contador("error", f"{valor:.0%}", C.error if valor > 0 else C.activacion)


def aciertos_de_recta(nube: NubePuntos, pendiente: float, corte: float) -> np.ndarray:
    """True donde la recta (en coordenadas de datos) clasifica bien cada punto."""
    arriba = nube.xy[:, 1] > pendiente * (nube.xy[:, 0] - 0.5) + corte
    return arriba == (nube.clases == 0)


# --- escenas ---------------------------------------------------------------------------------

class GanchoInvernadero(EscenaNarrada):
    """s01 · La recta ya está fallando en el primer segundo; la red encuentra la frontera."""
    segmento = "s01"

    RECTAS = [(-0.5, 0.35), (-0.15, 0.1), (-0.85, 0.6)]   # (pendiente, corte) en coordenadas de datos

    def construct(self):
        nube = datos()
        ejes = EjesEtiquetados(nube, "Temperatura (°C)", "Humedad (%)", (18, 32), (40, 90)).centrar(nube)
        recta = FronteraDecision.recta(nube, *self.RECTAS[0])
        self.add(ejes, ejes.nota)
        self.play(FadeIn(nube, lag_ratio=0.01), Create(recta), run_time=self.hueco(maximo=0.8))

        # «Y se equivoca»: los días mal clasificados, en rojo, con el contador.
        self.hasta_marca("marcar_errores")
        colores = {i: p.get_color() for i, p in enumerate(nube)}
        ok = aciertos_de_recta(nube, *self.RECTAS[0])
        fallos = VGroup(*[p for p, bien in zip(nube, ok) if not bien])
        marcador = etiqueta_error(1 - ok.mean())
        self.play(fallos.animate.set_color(C.error), FadeIn(marcador), run_time=self.hueco(maximo=0.7))

        # «Cada punto es un día»: se resalta uno con sus dos valores.
        self.hasta_marca("resaltar_dia")
        punto = nube[int(np.argmax(nube.xy[:, 0]))]
        halo = Circle(radius=0.22, color=C.atencion, stroke_width=3).move_to(punto)
        lectura = Text("27 °C · 78 %", font=FUENTE, font_size=TAM_NOTA, color=C.atencion)
        lectura.next_to(halo, UP, buff=0.2)
        self.play(Create(halo), FadeIn(lectura), run_time=self.hueco(maximo=0.6))

        # «Si el hongo apareció»: leyenda de las dos clases.
        self.hasta_marca("leyenda_clases")
        leyenda = VGroup(
            VGroup(Circle(radius=0.08, color=CLASES[1], fill_opacity=1),
                   Text("apareció el hongo", font=FUENTE, font_size=TAM_NOTA, color=C.texto_secundario)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(Circle(radius=0.08, color=CLASES[0], fill_opacity=1),
                   Text("no apareció", font=FUENTE, font_size=TAM_NOTA, color=C.texto_secundario)
                   ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(DOWN + RIGHT, buff=0.7)
        self.play(FadeIn(leyenda, shift=UP * 0.15), run_time=self.hueco(maximo=0.5))

        # «Más riesgo»: la zona de alerta por encima de la recta.
        self.hasta_marca("zona_alerta")
        zona = Polygon(recta.get_start(), recta.get_end(), recta.get_end() + UP * 3.2,
                       recta.get_start() + UP * 3.2, stroke_width=0, fill_color=C.error, fill_opacity=0.12)
        self.play(FadeIn(zona), FadeOut(halo, lectura), run_time=self.hueco(maximo=0.5))

        # «Da igual cómo la muevas»: otras inclinaciones, con el error actualizándose.
        self.hasta_marca("probar_rectas")
        self.play(FadeOut(zona), run_time=self.hueco(maximo=0.2))
        for pendiente, corte in self.RECTAS[1:]:
            ok = aciertos_de_recta(nube, pendiente, corte)
            self.play(Transform(recta, FronteraDecision.recta(nube, pendiente, corte)),
                      *[p.animate.set_color(C.error if not bien else colores[k])
                        for k, (p, bien) in enumerate(zip(nube, ok))],
                      Transform(marcador, etiqueta_error(1 - ok.mean())), run_time=self.hueco(maximo=0.8))

        self.hasta_marca("contador_final")
        self.play(Indicate(marcador, color=C.error, scale_factor=1.2), run_time=self.hueco(maximo=0.6))

        # «Esta frontera sí acierta»: la que aprendió la red.
        self.hasta_marca("frontera_empieza")
        self.play(Transform(recta, curva_frontera()),
                  *[p.animate.set_color(colores[k]) for k, p in enumerate(nube)],
                  Transform(marcador, etiqueta_error(0.0)),
                  run_time=self.tiempo_final(maximo=1.6))
        self.completar()


class LimiteLineal(EscenaNarrada):
    """s02 · Cortes rectos, clases entrelazadas y una fórmula a mano que no se reutiliza."""
    segmento = "s02"

    def construct(self):
        nube = datos(n=120)
        recta = FronteraDecision.recta(nube, -0.5, 0.35)
        panel = VGroup(nube, recta).scale(0.6).to_edge(LEFT, buff=1.2)
        self.add(panel)

        self.hasta_marca("corte_recto")
        self.play(Indicate(recta, color=C.texto, scale_factor=1.05), run_time=self.hueco(maximo=0.6))

        # «Cuando los dos grupos se entrelazan»: se señala la zona mezclada.
        self.hasta_marca("zoom_mezcla")
        mezcla = Circle(radius=0.9, color=C.atencion, stroke_width=3).move_to(panel.get_center())
        self.play(Create(mezcla), run_time=self.hueco(maximo=0.6))

        # «Inventar a mano una fórmula».
        self.hasta_marca("formula_a_mano")
        formula = MathTex(r"y > 0.4\,\sin(3x) + 0.2", color=C.texto_secundario, font_size=40)
        formula.next_to(panel, DOWN, buff=0.5)
        self.play(Write(formula), FadeOut(mezcla), run_time=self.hueco(maximo=0.8))

        # «En el de al lado»: primero uno, y con "curva" los otros dos.
        miniaturas = VGroup()
        for semilla in (1, 2, 3):
            mini = datos(n=90, semilla=semilla)
            miniaturas.add(VGroup(mini, FronteraDecision.curva(mini)).scale(0.32))
        miniaturas.arrange(DOWN, buff=0.4).to_edge(RIGHT, buff=1.4)
        self.hasta_marca("segundo_invernadero")
        self.play(FadeIn(miniaturas[0]), run_time=self.hueco(maximo=0.6))
        self.hasta_marca("mostrar_curvas")
        self.play(FadeIn(miniaturas[1], miniaturas[2], lag_ratio=0.4), run_time=self.hueco(maximo=1.0))

        self.hasta_marca("pregunta_final")
        pregunta = texto("¿Y si la encontrara sola?", tam=TAM_TEXTO, color=C.atencion)
        pregunta.next_to(panel, DOWN, buff=1.4)
        self.play(FadeOut(formula), Write(pregunta), run_time=self.tiempo_final(maximo=1.2))
        self.completar()


class NeuronaPasoAPaso(EscenaNarrada):
    """s03 · Cuerpo, entradas, pesos, suma, sesgo y activación, cada uno en su palabra."""
    segmento = "s03"

    def construct(self):
        neurona = Neurona()
        self.hasta_marca("aparece_neurona")
        self.play(neurona.aparecer_cuerpo(), run_time=self.hueco(maximo=0.7))
        self.hasta_marca("entran_valores")
        self.play(neurona.entrar_valores(), run_time=self.hueco(maximo=1.0))
        self.hasta_marca("mostrar_pesos")
        self.play(neurona.mostrar_pesos(), run_time=self.hueco(maximo=0.9))
        self.hasta_marca("resaltar_peso_mayor")
        self.play(Indicate(neurona.pesos[0], color=C.pesos, scale_factor=1.3), run_time=self.hueco(maximo=0.6))
        self.hasta_marca("mostrar_suma")
        self.play(neurona.mostrar_suma(), run_time=self.tiempo_hasta("mostrar_sesgo", maximo=0.8))
        self.hasta_marca("mostrar_sesgo")
        self.play(neurona.mostrar_sesgo(), run_time=self.hueco(maximo=0.6))
        # «Lo predispuesta que está a dispararse»: el cuerpo se enciende.
        self.hasta_marca("encender_cuerpo")
        self.play(neurona.cuerpo.animate.set_fill(C.activacion, opacity=0.25), run_time=self.hueco(maximo=0.4))
        self.play(neurona.cuerpo.animate.set_fill(C.panel, opacity=1.0), run_time=self.hueco(maximo=0.3))
        self.hasta_marca("mostrar_relu")
        self.play(neurona.mostrar_activacion(), run_time=self.hueco(maximo=1.0))
        # «Si el número es negativo, lo deja en cero»: la entrada negativa se apaga.
        self.hasta_marca("aplastar_negativos")
        self.play(neurona.destacar_relu(), neurona.valores[1].animate.set_color(C.texto_secundario),
                  run_time=self.tiempo_final(maximo=1.0))
        self.completar()


class CapaDeformaEspacio(EscenaNarrada):
    """s04 · La idea central: la capa deforma el espacio (pesos reales de la primera capa)."""
    segmento = "s04"

    def construct(self):
        # «Muchas neuronas juntas»: primero la capa como diagrama.
        capa = RedDensa([1, 5, 1]).scale(0.85)
        self.hasta_marca("formar_capa")
        self.play(capa.aparecer(), run_time=self.hueco(maximo=1.0))

        self.hasta_marca("enunciar_idea")
        idea = texto("Una capa deforma el espacio", tam=TAM_TEXTO, color=C.atencion).to_edge(UP, buff=0.6)
        self.play(FadeIn(idea, shift=DOWN * 0.15), run_time=self.hueco(maximo=0.6))

        nube = datos(n=150)
        rejilla = RejillaDeformable(nube, extension=4.5)
        self.hasta_marca("mostrar_rejilla")
        self.play(FadeOut(capa), run_time=self.hueco(maximo=0.4))
        self.play(FadeIn(rejilla), run_time=self.hueco(maximo=0.6))

        self.hasta_marca("estirar")
        self.play(rejilla.animar(W=DOS_CAPAS["W1"]), run_time=self.tiempo_hasta("desplazar", maximo=1.6))
        self.hasta_marca("desplazar")
        self.play(rejilla.animar(b=DOS_CAPAS["b1"]), run_time=self.tiempo_hasta("doblar", maximo=1.2))
        self.hasta_marca("doblar")
        self.play(rejilla.animar(relu=True), run_time=self.hueco(maximo=1.5))
        # «Aplasta contra cero todo lo negativo»: se señala el cuadrante que acaba de colapsar.
        self.hasta_marca("marcar_negativos")
        negativos = Polygon([-8, 0, 0], [0, 0, 0], [0, -5, 0], [-8, -5, 0],
                            stroke_width=0, fill_color=C.error, fill_opacity=0.10)
        self.play(FadeIn(negativos), run_time=self.hueco(maximo=0.4))
        self.play(FadeOut(negativos), run_time=0.3)

        # «Es el tapiz de debajo el que se reordena»: la cuadrícula se resalta y vuelve.
        self.hasta_marca("resaltar_tapiz")
        self.play(rejilla._lineas.animate.set_stroke(color=C.atencion, opacity=1.0), run_time=self.hueco(maximo=0.4))
        self.play(rejilla._lineas.animate.set_stroke(color=C.linea, opacity=0.6),
                  run_time=self.tiempo_final(maximo=0.6))
        self.completar()


class ApilarCapas(EscenaNarrada):
    """s05 · Una capa se queda al 10 % de error; con dos, cero."""
    segmento = "s05"

    def construct(self):
        nube = datos(n=150)
        rejilla = RejillaDeformable(nube, extension=4.5)
        W1, b1 = np.array(DOS_CAPAS["W1"]), np.array(DOS_CAPAS["b1"])
        rejilla.W, rejilla.b, rejilla.plegado = W1, b1, 1.0
        rejilla.escala_vista, rejilla.centro_vista = rejilla.encuadre(rejilla._nube_origen, W1, b1, 1.0)
        rejilla._actualizar()
        self.add(rejilla)

        self.hasta_marca("una_capa")
        rotulo = contador("capas", "1", C.texto_secundario).shift(DOWN * 1.1)
        self.play(FadeIn(rotulo), run_time=self.hueco(maximo=0.4))

        # «Diez arranques distintos»: diez marcas que se encienden (antes del "diez por ciento").
        self.hasta_marca("mostrar_intentos")
        intentos = VGroup(*[Circle(radius=0.07, color=C.texto_secundario, fill_opacity=1) for _ in range(10)])
        intentos.arrange(RIGHT, buff=0.16).next_to(ORIGIN, DOWN, buff=2.4)
        self.play(FadeIn(intentos, lag_ratio=0.15), run_time=self.hueco(maximo=0.8))

        self.hasta_marca("error_una_capa")
        error = etiqueta_error(UNA_CAPA["error"])
        self.play(FadeIn(error), run_time=self.hueco(maximo=0.5))

        # La segunda capa actúa sobre lo que dejó la primera (sin el encuadre, que es de pantalla).
        self.hasta_marca("segunda_capa")
        sin_vista = dict(escala_vista=1.0, centro_vista=np.zeros(2))
        rejilla._origen = [rejilla.transformar(p, **sin_vista) for p in rejilla._origen]
        rejilla._nube_origen = rejilla.transformar(rejilla._nube_origen, **sin_vista)
        rejilla.W, rejilla.b, rejilla.plegado = np.eye(2), np.zeros(2), 0.0
        self.play(rejilla.animar(W=np.array(DOS_CAPAS["W2"]), b=np.array(DOS_CAPAS["b2"]), relu=True),
                  Transform(rotulo, contador("capas", "2", C.texto_secundario).shift(DOWN * 1.1)),
                  FadeOut(intentos), run_time=self.hueco(maximo=1.6))

        self.hasta_marca("error_cero")
        self.play(Transform(error, etiqueta_error(DOS_CAPAS["error"])), run_time=self.hueco(maximo=0.5))

        self.hasta_marca("recta_final")
        v, c = np.array(DOS_CAPAS["v"]), float(DOS_CAPAS["c"][0])
        escala, centro = rejilla.escala_vista, rejilla.centro_vista
        recta = ParametricFunction(
            lambda t: np.array([*(np.array([t, (-c - v[0] * t) / v[1]]) - centro) * escala, 0]),
            t_range=(-1, 8, 0.1), color=C.texto, stroke_width=5)
        self.play(Create(recta), run_time=self.hueco(maximo=0.8))

        self.hasta_marca("resaltar_recta")
        self.play(Indicate(recta, color=C.activacion, scale_factor=1.02), run_time=self.hueco(maximo=0.6))

        self.hasta_marca("volver_al_origen")
        self.play(FadeOut(rejilla, recta, error, rotulo), FadeIn(datos(n=150), curva_frontera()),
                  run_time=self.tiempo_final(maximo=1.6))
        self.completar()


class PaisajeYGradiente(EscenaNarrada3D):
    """s06 · Pesos al azar, el error como paisaje y el descenso hasta el valle."""
    segmento = "s06"

    def construct(self):
        diagrama = RedDensa([2, 3, 3, 1]).scale(0.9)
        self.add_fixed_in_frame_mobjects(diagrama)
        self.play(diagrama.aparecer(), run_time=self.hueco(maximo=1.0))

        # «Empiezan al azar»: los pesos parpadean.
        self.hasta_marca("pesos_al_azar")
        for _ in range(2):
            self.play(diagrama.conexiones.animate.set_stroke(color=C.pesos, opacity=1.0), run_time=self.hueco(maximo=0.25))
            self.play(diagrama.conexiones.animate.set_stroke(color=C.linea, opacity=0.7), run_time=self.hueco(maximo=0.25))

        self.hasta_marca("mostrar_error")
        error = contador("error", "71 %", C.error)
        self.add_fixed_in_frame_mobjects(error)
        self.play(FadeIn(error), run_time=self.hueco(maximo=0.5))

        self.set_camera_orientation(phi=58 * DEGREES, theta=-50 * DEGREES, zoom=0.6)
        paisaje = PaisajePerdida().shift(DOWN * 0.6)
        self.hasta_marca("mostrar_paisaje")
        self.play(FadeOut(diagrama, error), run_time=self.hueco(maximo=0.4))
        self.remove_fixed_in_frame_mobjects(diagrama, error)
        self.play(Create(paisaje.superficie), run_time=self.hueco(maximo=1.2))
        self.add(paisaje.bola)

        # «Bajar la cuesta»: se marca el valle.
        self.hasta_marca("marcar_valle")
        fondo = min(((x, y) for x in np.linspace(-2, 2, 60) for y in np.linspace(-2, 2, 60)),
                    key=lambda p: paisaje.f(*p))
        senal = Circle(radius=0.3, color=C.activacion, stroke_width=4)
        senal.move_to(paisaje._punto(np.array(fondo)) + DOWN * 0.6).rotate(58 * DEGREES, axis=RIGHT)
        self.play(Create(senal), run_time=self.hueco(maximo=0.6))

        self.hasta_marca("mostrar_gradiente")
        g = paisaje.gradiente(paisaje.posicion)
        inicio = paisaje.bola.get_center()
        flecha = Arrow3D(inicio, inicio + np.array([*(g / np.linalg.norm(g)), 0.35]) * 0.9,
                         color=C.gradiente, thickness=0.02, base_radius=0.06)
        self.play(Create(flecha), run_time=self.hueco(maximo=0.6))

        self.hasta_marca("primer_paso")
        self.play(FadeOut(flecha), run_time=self.hueco(maximo=0.2))
        self.play(paisaje.descender(pasos=3), run_time=self.hueco(maximo=0.9))

        self.hasta_marca("descender")
        self.play(paisaje.descender(pasos=12), FadeOut(senal), run_time=self.tiempo_final(maximo=2.5))
        self.completar()


class MatematicasMinimas(EscenaNarrada):
    """s07 · Las dos fórmulas y qué pasa si la tasa de aprendizaje se pasa o se queda corta."""
    segmento = "s07"

    def construct(self):
        titulo = texto("Lo mismo, en corto", tam=TAM_TITULO).to_edge(UP, buff=0.7)
        self.hasta_marca("titulo_formulas")
        self.play(FadeIn(titulo), run_time=self.hueco(maximo=0.5))

        capa = MathTex(r"y", r"=", r"\sigma(", r"W", r"x", r"+", r"b", r")", font_size=60)
        capa.set_color_by_tex_to_color_map({r"W": C.pesos, r"b": C.pesos, r"x": C.datos, r"y": C.activacion})
        actualizacion = MathTex(r"w", r"\leftarrow", r"w", r"-", r"\eta", r"\frac{\partial L}{\partial w}",
                                font_size=60)
        actualizacion.set_color_by_tex_to_color_map({r"w": C.pesos, r"\eta": C.atencion,
                                                     r"\frac{\partial L}{\partial w}": C.gradiente})
        VGroup(capa, actualizacion).arrange(DOWN, buff=0.9).move_to(ORIGIN).shift(UP * 0.5)

        self.hasta_marca("formula_capa")
        self.play(Write(capa), run_time=self.tiempo_hasta("formula_actualizacion", maximo=1.5))
        # «Multiplica por los pesos, suma el sesgo»: se resaltan W y b.
        self.hasta_marca("resaltar_pesos")
        self.play(Indicate(capa.get_part_by_tex(r"W"), color=C.pesos, scale_factor=1.4),
                  Indicate(capa.get_part_by_tex(r"b"), color=C.pesos, scale_factor=1.4), run_time=self.hueco(maximo=0.8))

        self.hasta_marca("formula_actualizacion")
        self.play(Write(actualizacion), run_time=self.hueco(maximo=1.2))

        self.hasta_marca("resaltar_gradiente")
        self.play(Indicate(actualizacion.get_part_by_tex(r"\frac{\partial L}{\partial w}"),
                           color=C.gradiente, scale_factor=1.3), run_time=self.hueco(maximo=0.7))

        self.hasta_marca("resaltar_eta")
        eta = actualizacion.get_part_by_tex(r"\eta")
        nombre = Text("tasa de aprendizaje", font=FUENTE, font_size=TAM_NOTA, color=C.atencion)
        nombre.next_to(eta, DOWN, buff=0.45)
        self.play(Indicate(eta, color=C.atencion, scale_factor=1.4), FadeIn(nombre), run_time=self.hueco(maximo=0.8))

        # Dos valles: en uno la bola se pasa de largo; en el otro avanza lentísimo.
        perfil = lambda t: np.array([t, 0.35 * t ** 2, 0])
        izquierdo = ParametricFunction(perfil, t_range=(-1.9, 1.9, 0.05), color=C.error, stroke_width=4)
        derecho = izquierdo.copy().set_color(C.activacion)
        VGroup(izquierdo, derecho).arrange(RIGHT, buff=2.6).to_edge(DOWN, buff=0.6)
        bola_izq = Circle(radius=0.11, color=C.error, fill_opacity=1).move_to(izquierdo.get_start())
        bola_der = Circle(radius=0.11, color=C.activacion, fill_opacity=1).move_to(derecho.get_start())
        rotulo_izq = Text("η grande", font=FUENTE, font_size=TAM_NOTA, color=C.error)
        rotulo_izq.next_to(izquierdo, UP, buff=0.15)
        rotulo_der = Text("η pequeña", font=FUENTE, font_size=TAM_NOTA, color=C.activacion)
        rotulo_der.next_to(derecho, UP, buff=0.15)

        self.hasta_marca("paso_grande")
        self.play(FadeIn(izquierdo, bola_izq, rotulo_izq), run_time=self.hueco(maximo=0.4))
        for proporcion in (0.9, 0.1, 0.85):      # rebota de un lado a otro sin asentarse
            self.play(bola_izq.animate.move_to(izquierdo.point_from_proportion(proporcion)), run_time=self.hueco(maximo=0.25))

        self.hasta_marca("paso_pequeno")
        self.play(FadeIn(derecho, bola_der, rotulo_der), run_time=self.hueco(maximo=0.4))
        for proporcion in (0.07, 0.13, 0.19):    # avanza muy poco a poco
            self.play(bola_der.animate.move_to(derecho.point_from_proportion(proporcion)), run_time=self.hueco(maximo=0.25))
        self.completar()


class CuandoUsarla(EscenaNarrada):
    """s08 · Honestidad: en tablas pequeñas suele ganar gradient boosting."""
    segmento = "s08"

    def construct(self):
        tarjetas = TarjetaComparativa(
            usar=["Imágenes, audio, texto", "Muchos datos", "Patrones que se repiten"],
            evitar=["Tablas pequeñas", "Hay que explicar por qué", "Bastan reglas simples"])
        self.hasta_marca("mostrar_tarjetas")
        self.play(tarjetas.aparecer(), run_time=self.hueco(maximo=0.8))

        self.hasta_marca("tarjeta_evitar")
        self.play(tarjetas.rellenar("evitar"), run_time=self.hueco(maximo=1.2))

        self.hasta_marca("marcar_rapidez")
        sello = Text("entrena en segundos", font=FUENTE, font_size=TAM_NOTA, color=C.activacion)
        sello.next_to(tarjetas.paneles["evitar"], DOWN, buff=0.3)
        self.play(FadeIn(sello, shift=UP * 0.15), run_time=self.hueco(maximo=0.5))

        self.hasta_marca("marcar_explicable")
        explicable = Text("y explica su decisión", font=FUENTE, font_size=TAM_NOTA, color=C.activacion)
        explicable.next_to(sello, DOWN, buff=0.15)
        self.play(FadeIn(explicable, shift=UP * 0.12), run_time=self.hueco(maximo=0.5))

        self.hasta_marca("tarjeta_usar")
        self.play(tarjetas.rellenar("usar"), run_time=self.hueco(maximo=1.2))

        self.hasta_marca("cierre_tarjetas")
        self.play(Indicate(tarjetas.paneles["usar"], color=C.activacion, scale_factor=1.03),
                  run_time=self.tiempo_final(maximo=1.0))
        self.completar()


class PuenteACNN(EscenaNarrada):
    """s09 · De dos sensores a una foto: 150.528 números sin estructura."""
    segmento = "s09"

    def construct(self):
        invernadero = VGroup(datos(n=120), curva_frontera())
        self.add(invernadero)

        self.hasta_marca("resumen_deformacion")
        self.play(Indicate(invernadero, color=C.activacion, scale_factor=1.05), run_time=self.hueco(maximo=0.8))

        # «Dos sensores se quedan cortos»: se apagan.
        self.hasta_marca("sensores_insuficientes")
        sensores = VGroup(
            Text("temperatura", font=FUENTE, font_size=TAM_NOTA, color=C.datos),
            Text("humedad", font=FUENTE, font_size=TAM_NOTA, color=C.datos),
        ).arrange(RIGHT, buff=1.2).to_edge(UP, buff=0.8)
        self.play(FadeIn(sensores), run_time=self.hueco(maximo=0.5))
        self.play(invernadero.animate.scale(0.4).to_corner(UP + LEFT, buff=0.7),
                  sensores.animate.set_color(C.texto_secundario), run_time=self.hueco(maximo=0.7))

        self.hasta_marca("mostrar_imagen")
        imagen = Matriz.imagen(14, 14, lado=0.3).shift(DOWN * 0.3)
        self.play(imagen.aparecer(), run_time=self.hueco(maximo=0.9))

        self.hasta_marca("medidas_imagen")
        medidas = Text("224 × 224", font=FUENTE, font_size=TAM_NOTA, color=C.texto_secundario)
        medidas.next_to(imagen, DOWN, buff=0.25)
        self.play(FadeIn(medidas), run_time=self.hueco(maximo=0.4))

        # «Más de ciento cincuenta mil números»: el contador sube.
        self.hasta_marca("contar_pixeles")
        numero = Integer(0, font_size=TAM_TEXTO, color=C.datos).next_to(imagen, RIGHT, buff=0.9)
        pie = Text("números", font=FUENTE, font_size=TAM_NOTA, color=C.texto_secundario)
        pie.next_to(numero, DOWN, buff=0.15)
        self.play(FadeIn(numero, pie), run_time=self.hueco(maximo=0.3))
        self.play(ChangeDecimalToValue(numero, 150528), run_time=self.hueco(maximo=1.2))

        self.hasta_marca("aplanar_imagen")
        self.play(imagen.aplanar(), FadeOut(invernadero, sensores, medidas), run_time=self.hueco(maximo=1.3))

        self.hasta_marca("pregunta_final")
        pregunta = texto("¿Y ahora qué?", tam=TAM_TITULO).next_to(imagen, DOWN, buff=1.2)
        self.play(Write(pregunta), run_time=self.tiempo_final(maximo=1.0))
        self.completar()
