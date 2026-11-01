"""Clase base de las escenas sincronizadas con la voz (§10.4).

Cada escena declara su segmento; el episodio se deduce del archivo donde está
definida (episodios/epXX_slug/escenas.py), así que no hay rutas escritas a mano.

    class GanchoFrontera(EscenaNarrada):
        segmento = "s01"

        def construct(self):
            ...
            self.hasta_marca("frontera_empieza")
            ...
            self.completar()
"""
import inspect
import json
import math
from pathlib import Path

from manim import Scene, ThreeDScene

from lib import config
from lib import estilo  # noqa: F401  (aplica el fondo de la serie)


class SincronizadaConVoz:
    """Sincronización con el audio del segmento. Se combina con Scene o con ThreeDScene."""

    segmento: str = ""   # p. ej. "s01"

    @property
    def dir_episodio(self) -> Path:
        return Path(inspect.getfile(type(self))).resolve().parent

    def setup(self):
        if not self.segmento:
            raise ValueError(f"{type(self).__name__} no define 'segmento'")
        ruta = self.dir_episodio / "build" / "audio" / f"{self.segmento}.json"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}: ejecuta antes la etapa e2 (TTS) de este episodio")
        self.meta = json.loads(ruta.read_text(encoding="utf-8"))
        self.add_sound(str(config.RAIZ / self.meta["audio"]))

    @property
    def t(self) -> float:
        return self.renderer.time

    # Retraso tolerado antes de avisar de que una animación llega tarde a su marca.
    TOLERANCIA_S = 0.15

    def marca(self, evento: str) -> float:
        if evento not in self.meta["marcas"]:
            raise KeyError(f"Marca {evento!r} no existe en {self.segmento}: {sorted(self.meta['marcas'])}")
        return self.meta["marcas"][evento]

    def hasta_marca(self, evento: str):
        """Espera hasta el instante en que la voz pronuncia la marca. Avisa si ya se ha pasado."""
        restante = self.marca(evento) - self.t
        if restante > 0:
            self.wait(restante)
        elif -restante > self.TOLERANCIA_S:
            print(f"[AVISO] {self.segmento}: «{evento}» llega {-restante:.2f} s tarde "
                  f"(marca {self.marca(evento):.2f} s, escena en {self.t:.2f} s)")

    def hueco(self, maximo: float = 1.0, minimo: float = 0.2, reserva: float = 0.1) -> float:
        """run_time para una animación que empieza ahora y debe terminar antes de la marca
        siguiente (la que venga, sin nombrarla). Es lo que mantiene el ritmo cuando el guion
        tiene muchas marcas seguidas. Nunca devuelve más de lo que queda de audio."""
        siguientes = [t for t in self.meta["marcas"].values() if t > self.t + 0.05]
        limite = min(siguientes) if siguientes else self.meta["duracion"] - self.PERMANENCIA_MIN_S
        return self._cabe(max(minimo, min(maximo, limite - self.t - reserva)))

    def _cabe(self, run_time: float) -> float:
        """Recorta una duración para que la animación no se salga del audio del segmento: si se
        pasara, el vídeo duraría más que la voz e invadiría el segmento siguiente.

        Se redondea a fotogramas completos hacia abajo: Manim redondea hacia arriba y, con muchas
        animaciones seguidas, ese medio fotograma de más acaba sumando.
        """
        from manim import config as manim_config

        fps = float(manim_config.frame_rate)
        disponible = self.meta["duracion"] - self.t - 1 / fps
        return max(1 / fps, math.floor(min(run_time, disponible) * fps) / fps)

    def tiempo_hasta(self, evento: str, maximo: float = 1.0, minimo: float = 0.2) -> float:
        """run_time para que una animación que empieza ahora termine antes de la marca `evento`
        (acotado entre `minimo` y `maximo`, y sin salirse del audio)."""
        return self._cabe(max(minimo, min(maximo, self.marca(evento) - self.t)))

    # Tiempo mínimo que la última animación debe quedar en pantalla antes de que acabe el segmento.
    PERMANENCIA_MIN_S = 0.5

    def tiempo_final(self, maximo: float = 1.5, minimo: float = 0.3) -> float:
        """run_time para la última animación: que termine dejando PERMANENCIA_MIN_S en pantalla
        antes del final del audio (acotado entre `minimo` y `maximo`)."""
        disponible = self.meta["duracion"] - self.t - self.PERMANENCIA_MIN_S
        return self._cabe(max(minimo, min(maximo, disponible)))

    def completar(self):
        """Rellena hasta el final del audio para que vídeo y voz midan lo mismo (en fotogramas
        completos: pasarse invadiría el segmento siguiente)."""
        from manim import config as manim_config

        fps = float(manim_config.frame_rate)
        restante = math.floor((self.meta["duracion"] - self.t) * fps) / fps
        if restante < self.PERMANENCIA_MIN_S:
            print(f"[AVISO] {self.segmento}: la última animación termina {max(restante, 0):.2f} s antes del final "
                  f"del audio; no da tiempo a verla (mueve su marca antes o añade una pausa al guion)")
        if restante > 0:
            self.wait(restante)


class EscenaNarrada(SincronizadaConVoz, Scene):
    """Escena 2D sincronizada con la voz (el caso normal)."""


class EscenaNarrada3D(SincronizadaConVoz, ThreeDScene):
    """Igual, pero con cámara 3D: para el paisaje de la función de pérdida."""
