"""Procesado de audio con numpy: recorte de silencios, sonoridad y remuestreo.

Se usa soundfile para leer y escribir: en torchaudio 2.11, load/save necesitan TorchCodec.
"""
import numpy as np
import pyloudnorm
import soxr


def recortar_silencios(wav: np.ndarray, sr: int, umbral_db: float = -45.0, margen_s: float = 0.03) -> np.ndarray:
    """Quita el silencio inicial y final (ventanas de 10 ms por debajo del umbral), dejando un margen."""
    ventana = max(1, int(sr * 0.01))
    n = len(wav) // ventana
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[: n * ventana].reshape(n, ventana) ** 2, axis=1) + 1e-12)
    activas = np.flatnonzero(20 * np.log10(rms) > umbral_db)
    if activas.size == 0:
        return wav
    margen = int(sr * margen_s)
    inicio = max(0, activas[0] * ventana - margen)
    fin = min(len(wav), (activas[-1] + 1) * ventana + margen)
    return wav[inicio:fin]


def normalizar_sonoridad(wav: np.ndarray, sr: int, lufs: float = -20.0, pico_db: float = -1.0) -> np.ndarray:
    """Lleva la frase a una sonoridad fija para que todas suenen igual de fuertes, sin pasar del pico."""
    medidor = pyloudnorm.Meter(sr, block_size=min(0.4, len(wav) / sr * 0.9))
    actual = medidor.integrated_loudness(wav)
    if np.isfinite(actual):
        wav = wav * 10 ** ((lufs - actual) / 20)
    pico = np.max(np.abs(wav)) if wav.size else 0.0
    limite = 10 ** (pico_db / 20)
    if pico > limite:
        wav = wav * (limite / pico)
    return wav.astype(np.float32)


def remuestrear(wav: np.ndarray, sr_origen: int, sr_destino: int) -> np.ndarray:
    if sr_origen == sr_destino:
        return wav.astype(np.float32)
    return soxr.resample(wav, sr_origen, sr_destino, quality="VHQ").astype(np.float32)


def _db_por_ventana(wav: np.ndarray, sr: int, ventana_s: float = 0.01) -> tuple[np.ndarray, int]:
    ventana = max(1, int(sr * ventana_s))
    n = len(wav) // ventana
    rms = np.sqrt(np.mean(wav[: n * ventana].reshape(n, ventana) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms), ventana


def cortar_tras_hueco(wav: np.ndarray, sr: int, inicio_voz: float, fin_voz: float,
                      umbral_db: float = -40.0, hueco_s: float = 0.25) -> tuple[np.ndarray, float, str]:
    """Quita sonidos sueltos separados de la voz por un silencio (el "so" que el TTS añade a veces
    tras la frase). Solo mira fuera de [inicio_voz, fin_voz], la zona con palabras según Whisper.

    Devuelve (audio, segundos quitados al principio, descripción o "")."""
    db, ventana = _db_por_ventana(wav, sr)
    sonoro = db > umbral_db
    minimo = int(hueco_s / (ventana / sr))
    quitado_ini, partes = 0.0, []
    fin = len(wav)

    # Cola: primer hueco largo después de la última palabra, con sonido detrás.
    k = int(fin_voz * sr / ventana)
    while k + minimo <= len(sonoro):
        if not sonoro[k:k + minimo].any():
            if sonoro[k + minimo:].any():
                fin = min(len(wav), k * ventana + int(0.08 * sr))
                partes.append(f"cola {len(wav) / sr - fin / sr:.2f} s tras silencio")
            break
        k += 1

    # Cabeza: último hueco largo antes de la primera palabra, con sonido delante.
    inicio = 0
    k = int(inicio_voz * sr / ventana)
    while k - minimo >= 0:
        if not sonoro[k - minimo:k].any():
            if sonoro[:k - minimo].any():
                inicio = max(0, k * ventana - int(0.05 * sr))
                quitado_ini = inicio / sr
                partes.append(f"cabeza {quitado_ini:.2f} s antes de silencio")
            break
        k -= 1

    if not partes:
        return wav, 0.0, ""
    return fundidos(wav[inicio:fin], sr), quitado_ini, ", ".join(partes)


def fundidos(wav: np.ndarray, sr: int, segundos: float = 0.015) -> np.ndarray:
    """Fundido de entrada y salida muy corto para que un corte no suene como un chasquido."""
    n = min(int(sr * segundos), len(wav) // 2)
    if n == 0:
        return wav
    wav = wav.copy()
    rampa = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav[:n] *= rampa
    wav[-n:] *= rampa[::-1]
    return wav


def silencio(segundos: float, sr: int) -> np.ndarray:
    return np.zeros(int(round(segundos * sr)), dtype=np.float32)
