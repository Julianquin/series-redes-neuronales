"""Cuadrícula que se deforma como lo hace una capa: pesos (estirar y girar), sesgo (desplazar)
y activación ReLU (plegar contra los ejes).

Es la idea central del episodio 00: la capa no dibuja curvas, deforma el espacio.

    rejilla = RejillaDeformable(nube)          # la nube viaja con la cuadrícula
    self.play(rejilla.animar(W=[[1.4, -0.8], [0.6, 1.1]]), run_time=2)   # pesos
    self.play(rejilla.animar(b=[0.5, -0.3]))                             # sesgo
    self.play(rejilla.animar(relu=True))                                 # activación
"""
import numpy as np
from manim import Line, UpdateFromAlphaFunc, VGroup

from lib.estilo import C


class RejillaDeformable(VGroup):
    """Cuadrícula de líneas finas y, opcionalmente, una nube de puntos que sufre la misma
    transformación: x → ReLU(W·x + b), con el plegado aplicado de forma gradual.

    Cada animación parte del estado en que quedó la anterior.
    """

    def __init__(self, nube=None, extension: float = 4.0, paso: float = 0.5, muestras: int = 80, **kw):
        super().__init__(**kw)
        self.nube = nube
        self.W = np.eye(2)          # pesos
        self.b = np.zeros(2)        # sesgo
        self.plegado = 0.0          # 0 = sin ReLU, 1 = ReLU completa
        # Encuadre: una deformación con pesos reales agranda el espacio y se sale de pantalla,
        # así que la vista se reajusta con la misma animación.
        self.escala_vista = 1.0
        self.centro_vista = np.zeros(2)

        self._origen = []           # puntos (x, y) de cada línea, sin deformar
        t = np.linspace(-extension, extension, muestras)
        for v in np.arange(-extension, extension + paso / 2, paso):
            self._origen.append(np.column_stack([t, np.full_like(t, v)]))
            self._origen.append(np.column_stack([np.full_like(t, v), t]))
        self._lineas = VGroup()
        for _ in self._origen:
            self._lineas.add(Line(stroke_width=1.5, color=C.linea, stroke_opacity=0.6))
        self.add(self._lineas)
        if nube is not None:
            self._nube_origen = np.array([p.get_center()[:2] for p in nube])
            self.add(nube)
        self._actualizar()

    def transformar(self, xy: np.ndarray, W=None, b=None, plegado=None,
                    escala_vista=None, centro_vista=None) -> np.ndarray:
        """x → W·x + b, ReLU en la proporción `plegado` (0 a 1), y encuadre para la pantalla."""
        W = self.W if W is None else W
        b = self.b if b is None else b
        plegado = self.plegado if plegado is None else plegado
        escala = self.escala_vista if escala_vista is None else escala_vista
        centro = self.centro_vista if centro_vista is None else centro_vista
        y = xy @ W.T + b
        y = y * (1 - plegado) + np.maximum(y, 0) * plegado
        return (y - centro) * escala

    def encuadre(self, xy: np.ndarray, W=None, b=None, plegado=None, alto: float = 5.0):
        """(escala, centro) para que esos puntos, ya transformados, quepan en pantalla."""
        y = self.transformar(xy, W, b, plegado, escala_vista=1.0, centro_vista=np.zeros(2))
        centro = (y.max(0) + y.min(0)) / 2
        tamano = max((y.max(0) - y.min(0)).max(), 1e-6)
        return float(min(1.0, alto / tamano)), centro

    def _actualizar(self, W=None, b=None, plegado=None, escala_vista=None, centro_vista=None):
        for linea, pts in zip(self._lineas, self._origen):
            linea.set_points_as_corners(
                [[x, y, 0] for x, y in self.transformar(pts, W, b, plegado, escala_vista, centro_vista)])
        if self.nube is not None:
            for punto, (x, y) in zip(self.nube, self.transformar(self._nube_origen, W, b, plegado,
                                                                 escala_vista, centro_vista)):
                punto.move_to([x, y, 0])

    def animar(self, W=None, b=None, relu: bool | None = None, encuadrar: bool = True, **kw):
        """Anima hasta un estado nuevo. `W` y `b` sustituyen a los actuales; `relu=True` pliega
        (y `relu=False` despliega). Con `encuadrar`, la vista se reajusta a la vez para que la
        nube siga cabiendo en pantalla. Lo que no se indica se queda como está."""
        destino_W = np.array(W, dtype=float) if W is not None else self.W
        destino_b = np.array(b, dtype=float) if b is not None else self.b
        destino_plegado = self.plegado if relu is None else float(relu)
        if encuadrar and self.nube is not None:
            destino_escala, destino_centro = self.encuadre(self._nube_origen, destino_W, destino_b,
                                                           destino_plegado)
        else:
            destino_escala, destino_centro = self.escala_vista, self.centro_vista
        inicio = (self.W.copy(), self.b.copy(), self.plegado, self.escala_vista, self.centro_vista.copy())

        def paso(_, alpha: float):
            W_a = inicio[0] + (destino_W - inicio[0]) * alpha
            b_a = inicio[1] + (destino_b - inicio[1]) * alpha
            plegado_a = inicio[2] + (destino_plegado - inicio[2]) * alpha
            escala_a = inicio[3] + (destino_escala - inicio[3]) * alpha
            centro_a = inicio[4] + (destino_centro - inicio[4]) * alpha
            self._actualizar(W_a, b_a, plegado_a, escala_a, centro_a)
            if alpha >= 1.0:        # el estado final queda guardado para la siguiente animación
                self.W, self.b, self.plegado = destino_W, destino_b, destino_plegado
                self.escala_vista, self.centro_vista = destino_escala, destino_centro

        return UpdateFromAlphaFunc(self, paso, **kw)
