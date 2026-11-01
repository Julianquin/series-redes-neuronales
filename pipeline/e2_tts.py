"""Etapa 2 — Voz: TTS por frase, QA con Whisper y marcas (§10.3).

Por cada segmento del guion:
  1. divide la narración en frases y aplica pronunciación y cifras en palabras;
  2. sintetiza cada frase con el motor de config/voz.yaml y la estira a `velocidad`;
  3. la transcribe con faster-whisper: WER frente al texto y duración por carácter;
     si falla, la regenera con otra semilla (hasta voz_qa.intentos_max);
  4. recorta silencios, iguala sonoridad, concatena con las pausas y remuestrea a 48 kHz;
  5. escribe build/audio/<seg>.wav y <seg>.json (frases, marcas y QA).

Caché por frase: si ni el texto ni la voz ni los parámetros cambian, no se regenera.
TTS y Whisper comparten proceso a propósito (§13.3: caben juntos) para poder reintentar.

    python -m pipeline.e2_tts --ep 99 [--segmento s02] [--simulado]

Tras escuchar una frase marcada para revisión:
    --aceptar s01:f02      la das por buena (queda registrada como aprobada a mano)
    --regenerar s01:f02    nuevos intentos con semillas que no se han probado todavía
"""
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Las descargas por Xet se han colgado dos veces en esta máquina: se usa HTTP clásico.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _sin_conexion(motor: str) -> None:
    """Con los modelos ya descargados, Chatterbox y Whisper trabajan sin conexión: sin avisos del
    Hub y sin depender de él. Qwen3-TTS no: su cargador consulta la API y el modo sin conexión lo
    hace fallar en vez de usar la caché. Se llama antes de importar nada de huggingface_hub."""
    if motor != "chatterbox_multilingual":
        return
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    necesarios = ("models--ResembleAI--chatterbox", "models--Systran--faster-whisper-large-v3")
    if all((hub / m).is_dir() for m in necesarios):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Sin barras de progreso por frase ni avisos internos de Chatterbox: el log solo lleva el QA.
os.environ.setdefault("TQDM_DISABLE", "1")

import logging  # noqa: E402
import warnings  # noqa: E402

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
logging.getLogger("chatterbox").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lib import audio, config  # noqa: E402
from lib.texto import (  # noqa: E402
    Frase, dividir_frases, normalizar_para_wer, numeros_a_palabras, terminos_pronunciacion, texto_para_tts,
)
from pipeline import comun  # noqa: E402


@dataclass
class ResultadoFrase:
    clave: str
    texto: str
    texto_tts: str
    duracion: float
    wer: float
    s_por_caracter: float
    transcripcion: str
    palabras: list[dict]          # [{"palabra", "inicio", "fin"}] relativos a la frase recortada
    intentos: int = 1
    estado: str = "ok"            # ok | revisar | aceptada (a mano, tras escucharla) | simulado
    problemas: list[str] = field(default_factory=list)
    recorte: str = ""             # sonidos sueltos eliminados ("cola 1.44 s tras silencio")
    motor: str = ""               # con qué motor se generó (el ritmo se mide por motor)


# Versión del procesado de audio: forma parte de la clave de caché, así que subirla
# regenera las frases cuando cambia la forma de producirlas.
PROCESADO = 4   # 4: la velocidad del narrador se aplica antes del QA


# --- Modelos (se cargan solo si alguna frase necesita generarse) ---------------------------

class Voz:
    """Chatterbox Multilingual: clona la voz solo con el audio de referencia."""

    def __init__(self, voz_cfg: dict):
        self.cfg = voz_cfg
        self.ajustes = voz_cfg["chatterbox"]
        self._modelo = None

    def _cargar(self):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        print("Cargando Chatterbox Multilingual…", flush=True)
        self._torch = torch
        self._modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self._modelo.prepare_conditionals(str(config.RAIZ / self.cfg["referencia"]),
                                          exaggeration=self.ajustes["parametros"]["exaggeration"])

    def sintetizar(self, texto: str, semilla: int) -> tuple[np.ndarray, int]:
        if self._modelo is None:
            self._cargar()
        self._torch.manual_seed(semilla)
        wav = self._modelo.generate(texto, language_id=self.cfg["idioma"], **self.ajustes["parametros"])
        return wav.squeeze(0).cpu().numpy().astype(np.float32), self._modelo.sr


class VozQwen:
    """Qwen3-TTS: necesita el audio de referencia y su transcripción (config/voz.yaml).

    Vive en su propio venv (.venv-qwen) porque pide otra versión de transformers, así que la
    etapa 2 entera se ejecuta con ese intérprete (voz.python en la configuración).
    """

    def __init__(self, voz_cfg: dict):
        self.cfg = voz_cfg
        self.ajustes = voz_cfg["qwen"]
        self._modelo = None

    def _cargar(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        print(f"Cargando {self.ajustes['modelo']}…", flush=True)
        self._torch = torch
        self._modelo = Qwen3TTSModel.from_pretrained(self.ajustes["modelo"], device_map="cuda:0",
                                                     dtype=torch.bfloat16)

    def sintetizar(self, texto: str, semilla: int) -> tuple[np.ndarray, int]:
        if self._modelo is None:
            self._cargar()
        self._torch.manual_seed(semilla)
        wavs, sr = self._modelo.generate_voice_clone(
            text=texto, language=self.ajustes.get("idioma", "Spanish"),
            ref_audio=str(config.RAIZ / self.cfg["referencia"]),
            ref_text=self.cfg["texto_referencia"],
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)


MOTORES = {"chatterbox_multilingual": Voz, "qwen3_tts": VozQwen}


def crear_voz(voz_cfg: dict):
    motor = voz_cfg["motor"]
    if motor not in MOTORES:
        raise ValueError(f"Motor de voz desconocido: {motor} (hay {', '.join(MOTORES)})")
    return MOTORES[motor](voz_cfg)


class Oido:
    def __init__(self, modelo: str):
        self.nombre = modelo
        self._modelo = None

    def transcribir(self, ruta: Path) -> tuple[str, list[dict]]:
        if self._modelo is None:
            from faster_whisper import WhisperModel

            print(f"Cargando Whisper {self.nombre}…", flush=True)
            self._modelo = WhisperModel(self.nombre, device="cuda", compute_type="float16")
        segmentos, _ = self._modelo.transcribe(str(ruta), language="es", beam_size=5, word_timestamps=True)
        palabras = [{"palabra": w.word.strip(), "inicio": round(w.start, 3), "fin": round(w.end, 3)}
                    for s in segmentos for w in (s.words or [])]
        return " ".join(p["palabra"] for p in palabras), palabras


class VozSimulada:
    """Tono en lugar de voz: ~0,07 s por carácter. Para probar el pipeline sin GPU."""
    def sintetizar(self, texto: str, semilla: int) -> tuple[np.ndarray, int]:
        sr = 24000
        t = np.arange(int(len(texto) * 0.07 * sr)) / sr
        return (0.2 * np.sin(2 * np.pi * (180 + semilla % 40) * t)).astype(np.float32), sr


class OidoSimulado:
    """Devuelve el texto original con tiempos repartidos por longitud de palabra."""
    def __init__(self):
        self.texto = ""

    def transcribir(self, ruta: Path) -> tuple[str, list[dict]]:
        duracion = sf.info(str(ruta)).duration
        tokens = self.texto.split()
        total = sum(len(t) + 1 for t in tokens)
        palabras, t = [], 0.0
        for tok in tokens:
            d = duracion * (len(tok) + 1) / total
            palabras.append({"palabra": tok, "inicio": round(t, 3), "fin": round(t + d, 3)})
            t += d
        return self.texto, palabras


# --- Frases ---------------------------------------------------------------------------------

def clave_frase(texto_tts: str, voz_cfg: dict, huella_ref: str, modelo_whisper: str, simulado: bool) -> str:
    motor = voz_cfg["motor"]
    datos = {
        "texto": texto_tts, "motor": motor, "ajustes": voz_cfg.get(motor.split("_")[0], {}),
        "semilla": voz_cfg["semilla"], "idioma": voz_cfg["idioma"], "referencia": huella_ref,
        "whisper": modelo_whisper, "simulado": simulado, "procesado": PROCESADO,
        "velocidad": voz_cfg.get("velocidad", 1.0),
    }
    return hashlib.sha256(json.dumps(datos, sort_keys=True).encode()).hexdigest()[:16]


def evaluar(texto: str, transcripcion: str, duracion: float, qa: dict, excluir: set[str]) -> tuple[float, float, list[str]]:
    ref = normalizar_para_wer(texto, excluir)
    hip = normalizar_para_wer(transcripcion, excluir)
    wer = jiwer.wer(ref, hip) if ref else 0.0
    caracteres = max(1, len(numeros_a_palabras(texto)))
    s_car = duracion / caracteres
    problemas = []
    if wer > qa["wer_max_frase"]:
        problemas.append(f"WER {wer:.2f} > {qa['wer_max_frase']}")
    # Ritmo: se compara con lo que dura una frase de esa longitud en esta voz (ver lib/ritmo.py).
    a, b = qa["duracion_modelo"]
    esperado = a + b * caracteres
    ratio = duracion / esperado
    lo, hi = qa["duracion_ratio"]
    if not lo <= ratio <= hi:
        problemas.append(f"dura {duracion:.2f} s y se esperaban {esperado:.2f} s "
                         f"(ratio {ratio:.2f}, fuera de [{lo}, {hi}])")
    if s_car > qa["seg_por_caracter_max"]:
        problemas.append(f"{s_car:.3f} s/carácter: el TTS se ha enganchado")
    return wer, s_car, problemas


def detectar_sobrantes(frase: Frase, palabras: list[dict]) -> str:
    """Chatterbox a veces añade un "oh"/"eh" pegado a la voz, antes o después de la frase. Si Whisper
    oye palabras que no están en el guion antes de la primera o después de la última, lo describe.

    No se recorta: el sonido va fundido con la palabra y el corte deja restos audibles (ep99 s01_f02).
    El intento se descarta y se genera otro. Los sonidos separados por silencio sí se recortan
    (limpiar_bordes)."""
    ref, ref_tts = _tokens(frase.texto), _tokens(texto_para_tts(frase.texto))
    if not ref or not palabras:
        return ""
    toks = [_tokens(p["palabra"]) for p in palabras]
    n = len(palabras)
    primera = {ref[0], ref_tts[0] if ref_tts else ref[0]}
    ultima = {ref[-1], ref_tts[-1] if ref_tts else ref[-1]}
    i0 = next((i for i in range(min(3, n)) if primera & set(toks[i])), 0)
    j0 = next((j for j in range(n - 1, max(-1, n - 4), -1) if ultima & set(toks[j])), n - 1)
    sobra_ini = [palabras[i]["palabra"] for i in range(i0) if toks[i]]
    sobra_fin = [palabras[j]["palabra"] for j in range(j0 + 1, n) if toks[j]] if j0 >= i0 else []
    partes = ([f"inicio «{' '.join(sobra_ini)}»"] if sobra_ini else []) + \
             ([f"final «{' '.join(sobra_fin)}»"] if sobra_fin else [])
    return ", ".join(partes)


def problemas_de_bordes(r: "ResultadoFrase", frase: Frase) -> list[str]:
    """Vocalizaciones pegadas a la voz. Incluye las frases de versiones anteriores que se recortaron
    por palabras (método retirado: dejaba restos del "oh")."""
    problemas = []
    sobrantes = detectar_sobrantes(frase, r.palabras)
    if sobrantes:
        problemas.append(f"vocalización pegada a la voz: {sobrantes}")
    if re.search(r"(inicio|final) «", r.recorte):
        problemas.append("recortada por palabras con un método retirado")
    return problemas


def limpiar_bordes(wav: np.ndarray, sr: int, palabras: list[dict]) -> tuple[np.ndarray, float, str]:
    """Corta sonidos sueltos separados de la voz por un silencio, usando las palabras de Whisper
    como zona de voz. Devuelve (audio, segundos quitados al principio, descripción)."""
    if not palabras:
        return wav, 0.0, ""
    return audio.cortar_tras_hueco(wav, sr, palabras[0]["inicio"], palabras[-1]["fin"])


def limpiar_en_cache(r: "ResultadoFrase", ruta_wav: Path, k: int) -> None:
    """Aplica limpiar_bordes a una frase generada antes de que existiera, sin regenerarla.
    Es idempotente: una frase ya limpia no cambia."""
    wav, sr = sf.read(ruta_wav, dtype="float32")
    wav, quitado, cola = limpiar_bordes(wav, sr, r.palabras)
    if not cola:
        return
    print(f"  f{k:02d} recortada en caché: {cola}", flush=True)
    sf.write(ruta_wav, wav, sr)
    r.palabras = _desplazar(r.palabras, quitado)
    r.duracion = round(len(wav) / sr, 3)
    r.recorte = " + ".join(x for x in (r.recorte, cola) if x)
    r.s_por_caracter = round(r.duracion / max(1, len(numeros_a_palabras(r.texto))), 4)


def _desplazar(palabras: list[dict], segundos: float) -> list[dict]:
    if not segundos:
        return palabras
    return [dict(p, inicio=round(p["inicio"] - segundos, 3), fin=round(p["fin"] - segundos, 3)) for p in palabras]


def estirar(ruta: Path, velocidad: float) -> tuple[np.ndarray, int]:
    """Cambia la velocidad sin tocar el tono (ffmpeg atempo) y reescribe el wav.

    Por debajo de 1 habla más despacio. Se aplica antes de transcribir para que el QA, las marcas
    y la duración midan exactamente el audio que se va a publicar.
    """
    estirado = ruta.with_suffix(".atempo.wav")
    comun.ffmpeg("-i", str(ruta), "-filter:a", f"atempo={velocidad:.4f}", str(estirado))
    estirado.replace(ruta)
    return sf.read(ruta, dtype="float32")


def generar_frase(frase: Frase, ruta_wav: Path, clave: str, voz, oido, voz_cfg: dict, qa: dict,
                  excluir: set[str], simulado: bool, previo: "ResultadoFrase | None" = None) -> ResultadoFrase:
    """Genera la frase hasta que pasa el QA; devuelve el mejor intento.
    Con `previo` (--regenerar) usa semillas nuevas y conserva la versión anterior si ninguna la mejora."""
    desplazamiento = previo.intentos if previo else 0
    velocidad = float(voz_cfg.get("velocidad", 1.0))
    texto_tts = texto_para_tts(frase.texto)
    if isinstance(oido, OidoSimulado):
        oido.texto = frase.texto
    mejor: ResultadoFrase | None = previo
    for intento in range(1, qa["intentos_max"] + 1):
        semilla = voz_cfg["semilla"] + 1000 * (desplazamiento + intento - 1)
        wav, sr = voz.sintetizar(texto_tts, semilla)
        wav = audio.normalizar_sonoridad(audio.recortar_silencios(wav, sr), sr)
        tmp = ruta_wav.with_suffix(f".intento{intento}.wav")
        sf.write(tmp, wav, sr)
        if velocidad != 1.0:
            wav, sr = estirar(tmp, velocidad)
        transcripcion, palabras = oido.transcribir(tmp)
        # Sonidos sueltos separados de la voz por un silencio: se recortan (fiable).
        wav, quitado, recorte = limpiar_bordes(wav, sr, palabras)
        if recorte:
            palabras = _desplazar(palabras, quitado)
            sf.write(tmp, wav, sr)
        duracion = len(wav) / sr
        wer, s_car, problemas = evaluar(frase.texto, transcripcion, duracion, qa, excluir)
        r = ResultadoFrase(clave, frase.texto, texto_tts, round(duracion, 3), round(wer, 3), round(s_car, 4),
                           transcripcion, palabras, intento, "ok", problemas, recorte,
                           "simulado" if simulado else voz_cfg["motor"])
        # Vocalizaciones pegadas a la voz ("oh" + palabra): no se recortan, se descarta el intento.
        r.problemas += problemas_de_bordes(r, frase)
        r.estado = "simulado" if simulado else ("revisar" if r.problemas else "ok")
        problemas = r.problemas
        print(f"    intento {desplazamiento + intento}: WER {wer:.2f} · {s_car:.3f} s/car"
              + (f" · recortado {recorte}" if recorte else "")
              + (f" · {'; '.join(problemas)}" if problemas else ""), flush=True)
        if mejor is None or (len(r.problemas), r.wer) < (len(mejor.problemas), mejor.wer):
            mejor = r
            tmp.replace(ruta_wav)
        else:
            tmp.unlink()
        if not problemas:
            break
    mejor.intentos = desplazamiento + intento
    return mejor


# --- Marcas ---------------------------------------------------------------------------------

def _tokens(texto: str) -> list[str]:
    return normalizar_para_wer(texto).split()


def ubicar_marca(palabra: str, frases: list[Frase], resultados: list[ResultadoFrase],
                 inicios: list[float], ocurrencia: int = 1) -> tuple[float, str]:
    """Instante (s, relativo al segmento) en que se pronuncia `palabra` y método usado.
    `ocurrencia` elige qué aparición, cuando la palabra se dice más de una vez."""
    objetivo = _tokens(palabra)
    variantes = [objetivo, _tokens(texto_para_tts(palabra))]
    vistas = 0
    for frase, r, t0 in zip(frases, resultados, inicios):
        # Una misma frase puede contener la palabra varias veces: se cuentan todas.
        en_frase = _apariciones(_tokens(frase.texto), objetivo)
        if not en_frase:
            continue
        if vistas + len(en_frase) < ocurrencia:
            vistas += len(en_frase)
            continue
        salto = ocurrencia - vistas - 1      # cuántas apariciones saltar dentro de esta frase
        vistas += len(en_frase)
        dichas = [(_tokens(p["palabra"]), p["inicio"]) for p in r.palabras]
        planas = [(tok, ini) for toks, ini in dichas for tok in toks]
        seq = [tok for tok, _ in planas]
        for v in variantes:
            indices = [i for i in range(len(seq) - len(v) + 1) if seq[i:i + len(v)] == v]
            if len(indices) > salto:
                return round(t0 + planas[indices[salto]][1], 3), "whisper"
        # Whisper no la reconoció: se estima por su posición en el texto de la frase.
        pos = frase.texto.lower().find(palabra.lower())
        frac = max(0, pos) / max(1, len(frase.texto))
        return round(t0 + frac * r.duracion, 3), "estimada"
    raise ValueError(f"la marca «{palabra}» no aparece en ninguna frase")


def _indice(seq: list[str], sub: list[str]) -> int | None:
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return None


def _contiene(seq: list[str], sub: list[str]) -> bool:
    return _indice(seq, sub) is not None


def _apariciones(seq: list[str], sub: list[str]) -> list[int]:
    """Todas las posiciones de `sub` en `seq` (una frase puede repetir la palabra de la marca)."""
    return [i for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub]


# --- Segmentos ------------------------------------------------------------------------------

def procesar_segmento(seg: dict, dir_audio: Path, voz, oido, voz_cfg: dict, qa: dict, excluir: set[str],
                      huella_ref: str, simulado: bool, aceptar: set[str] = frozenset(),
                      regenerar: set[str] = frozenset()) -> dict:
    sid = seg["id"]
    frases = dividir_frases(seg["narracion"])
    dir_frases = dir_audio / "frases"
    dir_frases.mkdir(parents=True, exist_ok=True)
    print(f"{sid}: {len(frases)} frases", flush=True)

    resultados: list[ResultadoFrase] = []
    for k, frase in enumerate(frases, 1):
        base = dir_frases / f"{sid}_f{k:02d}"
        clave = clave_frase(texto_para_tts(frase.texto), voz_cfg, huella_ref, qa["modelo_whisper"], simulado)
        meta = base.with_suffix(".json")
        ref = f"{sid}:f{k:02d}"
        previo = None
        if meta.exists() and base.with_suffix(".wav").exists():
            previo = ResultadoFrase(**json.loads(meta.read_text(encoding="utf-8")))
            if previo.clave != clave:
                previo = None
        # Se reevalúa con los umbrales actuales: cambiar un umbral no obliga a regenerar audio.
        # Si una frase que estaba "ok" deja de pasar, se regenera sola; las que ya estaban en
        # "revisar" esperan a que decidas (--aceptar / --regenerar).
        forzar = ref in regenerar
        if previo:
            limpiar_en_cache(previo, base.with_suffix(".wav"), k)
        if previo and previo.estado in ("ok", "revisar"):
            estado_antes = previo.estado
            _, _, previo.problemas = evaluar(previo.texto, previo.transcripcion, previo.duracion, qa, excluir)
            previo.problemas += problemas_de_bordes(previo, frase)
            previo.estado = "revisar" if previo.problemas else "ok"
            if estado_antes == "ok" and previo.estado == "revisar":
                print(f"  f{k:02d} ya no pasa el QA actual ({'; '.join(previo.problemas)}): se regenera", flush=True)
                forzar = True
        if previo and not forzar:
            r = previo
            print(f"  f{k:02d} en caché ({r.estado})", flush=True)
        else:
            print(f"  f{k:02d} «{frase.texto}»", flush=True)
            r = generar_frase(frase, base.with_suffix(".wav"), clave, voz, oido, voz_cfg, qa, excluir, simulado,
                              previo)
        if ref in aceptar and r.estado == "revisar":
            r.estado = "aceptada"
            print(f"  f{k:02d} aceptada a mano ({'; '.join(r.problemas)})", flush=True)
        meta.write_text(json.dumps(asdict(r), ensure_ascii=False, indent=1), encoding="utf-8")
        resultados.append(r)

    # Concatenación: frase, pausa (general + [pausa:x]), frase, … a la frecuencia de salida.
    sr_out = voz_cfg["sample_rate_salida"]
    partes, inicios, t = [], [], 0.0
    lista_frases = []
    for k, (frase, r) in enumerate(zip(frases, resultados), 1):
        ruta_frase = dir_frases / f"{sid}_f{k:02d}.wav"
        wav, sr = sf.read(ruta_frase, dtype="float32")
        wav = audio.remuestrear(wav, sr, sr_out)
        inicios.append(t)
        partes.append(wav)
        fin = t + len(wav) / sr_out
        lista_frases.append({"texto": frase.texto, "inicio": round(t, 3), "fin": round(fin, 3),
                             "wer": r.wer, "intentos": r.intentos, "estado": r.estado})
        t = fin
        if k < len(frases) or frase.pausa_extra:
            pausa = (voz_cfg["pausa_entre_frases_s"] if k < len(frases) else 0) + frase.pausa_extra
            partes.append(audio.silencio(pausa, sr_out))
            t += len(partes[-1]) / sr_out
    ruta_wav = dir_audio / f"{sid}.wav"
    sf.write(ruta_wav, np.concatenate(partes), sr_out, subtype="PCM_16")

    marcas, metodos = {}, {}
    for m in seg.get("marcas") or []:
        marcas[m["evento"]], metodos[m["evento"]] = ubicar_marca(
            m["palabra"], frases, resultados, inicios, int(m.get("ocurrencia", 1)))
        if metodos[m["evento"]] == "estimada":
            print(f"  ⚠ marca «{m['palabra']}» estimada: Whisper no la reconoció", flush=True)

    estados = {r.estado for r in resultados}
    salida = {
        "segmento": sid,
        "audio": str(ruta_wav.relative_to(config.RAIZ)),
        "duracion": round(sf.info(str(ruta_wav)).duration, 3),
        "frases": lista_frases,
        "marcas": marcas,
        "marcas_metodo": metodos,
        "qa": {"wer": max(r.wer for r in resultados), "intentos": max(r.intentos for r in resultados),
               "estado": "revisar" if "revisar" in estados else ("simulado" if simulado else "ok")},
    }
    (dir_audio / f"{sid}.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    return salida


def _refs_frase(valor: str) -> set[str]:
    refs = {v.strip() for v in valor.split(",") if v.strip()}
    for r in refs:
        if not re.fullmatch(r"s\w+:f\d{2}", r):
            raise SystemExit(f"Referencia de frase no válida: {r!r} (formato: s01:f02)")
    return refs


def _argumentos_e2(p):
    p.add_argument("--aceptar", type=_refs_frase, default=set(), metavar="s01:f02[,…]",
                   help="dar por buenas frases marcadas para revisión, tras escucharlas")
    p.add_argument("--regenerar", type=_refs_frase, default=set(), metavar="s01:f02[,…]",
                   help="volver a generar frases con semillas nuevas")


def main() -> int:
    args = comun.argumentos(__doc__, _argumentos_e2)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    voz_cfg = config.cargar("voz")
    qa = config.cargar("pipeline")["voz_qa"]
    _sin_conexion(voz_cfg["motor"])   # antes de que se importe nada de huggingface_hub

    ref = config.RAIZ / voz_cfg["referencia"]
    if not ref.exists():
        print(f"[FALLO] falta la voz de referencia {ref} (ver notebooks/01_calibrar_voz.ipynb)")
        return comun.FALLO
    if voz_cfg["motor"] == "qwen3_tts" and not voz_cfg.get("texto_referencia"):
        print("[FALLO] Qwen3-TTS necesita la transcripción de la voz de referencia y config/voz.yaml "
              "la tiene vacía.\n   Ejecuta:  .venv/bin/python -m lib.voz_referencia")
        return comun.FALLO
    huella_ref = hashlib.sha256(ref.read_bytes()).hexdigest()
    calibrada = (voz_cfg.get("calibracion") or {}).get("sha256_referencia")
    if calibrada and calibrada != huella_ref:
        print("[AVISO] narrador.wav ha cambiado desde la calibración: conviene recalibrar la voz")

    segmentos = [s for s in guion["segmentos"] if not args.segmento or s["id"] == args.segmento]
    if not segmentos:
        print(f"[FALLO] no existe el segmento {args.segmento}")
        return comun.FALLO

    if args.simulado:
        print("[SIMULADO] voz y transcripción falsas: solo para probar el pipeline")
        voz, oido = VozSimulada(), OidoSimulado()
    else:
        voz, oido = crear_voz(voz_cfg), Oido(qa["modelo_whisper"])
    dir_audio = comun.dir_build(dir_ep, "audio")
    excluir = terminos_pronunciacion()

    revisar = []
    for seg in segmentos:
        salida = procesar_segmento(seg, dir_audio, voz, oido, voz_cfg, qa, excluir, huella_ref, args.simulado,
                                   args.aceptar, args.regenerar)
        print(f"  → {salida['duracion']:.2f} s · WER máx {salida['qa']['wer']:.2f} · {salida['qa']['estado']}", flush=True)
        revisar += [f"{seg['id']} «{f['texto']}»" for f in salida["frases"] if f["estado"] == "revisar"]

    if revisar:
        print("\n[FALLO] frases que no pasan el QA de voz tras todos los intentos (escúchalas en build/audio/frases/):")
        for r in revisar:
            print(f"  - {r}")
        print("Escúchalas y elige:\n"
              "  · suena bien  → python -m pipeline.e2_tts --ep <ep> --aceptar s01:f02\n"
              "  · suena mal   → python -m pipeline.e2_tts --ep <ep> --regenerar s01:f02\n"
              "  · falla siempre → reescribe la frase o añade el término a config/pronunciacion.yaml")
        return comun.FALLO
    return comun.OK


if __name__ == "__main__":
    sys.exit(main())
