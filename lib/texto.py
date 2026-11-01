"""Procesado del texto de narración: frases, pausas explícitas y pronunciación.

Lo usan la validación del guion (e1), el TTS (e2), los subtítulos (e7) y el
notebook de calibración de voz, para que todos partan exactamente del mismo corte.
"""
import re
import unicodedata
from dataclasses import dataclass
from functools import cache

from lib import config

_PAUSA = re.compile(r"\[pausa:\s*([0-9]*\.?[0-9]+)\s*\]")
# Corta en el espacio que sigue a . ! ? … (o a esos signos más comillas o paréntesis de cierre).
_FIN_FRASE = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"»”)]))\s+")


@dataclass
class Frase:
    texto: str               # texto original: subtítulos, WER y marcas
    pausa_extra: float = 0   # silencio adicional tras la frase ([pausa:x])

    @property
    def palabras(self) -> int:
        return len(self.texto.split())


def dividir_frases(narracion: str) -> list[Frase]:
    """Divide una narración en frases, asignando cada [pausa:x] a la frase anterior."""
    frases: list[Frase] = []
    trozos = _PAUSA.split(" ".join(narracion.split()))
    # re.split con un grupo alterna: texto, pausa, texto, pausa, ...
    for i, trozo in enumerate(trozos):
        if i % 2 == 1:
            if frases:
                frases[-1].pausa_extra += float(trozo)
            continue
        for parte in _FIN_FRASE.split(trozo.strip()):
            if parte.strip():
                frases.append(Frase(parte.strip()))
    return frases


@cache
def _patron_pronunciacion() -> tuple[re.Pattern, dict[str, str]] | None:
    dic = {k.lower(): v for k, v in config.cargar("pronunciacion").items()}
    if not dic:
        return None
    # Términos más largos primero para que "backpropagation" gane a un posible "back".
    terminos = sorted(dic, key=len, reverse=True)
    patron = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, terminos)) + r")(?!\w)", re.IGNORECASE)
    return patron, dic


_R_INICIAL = re.compile(r"(?<![\w-])([rR])(?=[aeiouáéíóúAEIOUÁÉÍÓÚ])(\w*)")


def erre_inicial(texto: str, respetar_diccionario: bool = True) -> str:
    """r inicial + vocal → rr ("red" → "rred"). En español suenan igual, y así el TTS no lee con
    acento inglés las palabras que existen en ambos idiomas. Respeta las siglas en mayúsculas y
    los términos del diccionario de pronunciación, que ya llevan su propia escritura."""
    terminos = terminos_pronunciacion() if respetar_diccionario else set()

    def sustituir(m: re.Match) -> str:
        palabra = m.group(0)
        if palabra.lower() in terminos or (palabra.isupper() and len(palabra) > 1) or any(c.isupper() for c in palabra[1:]):
            return palabra
        return m.group(1) + "r" + m.group(2)

    return _R_INICIAL.sub(sustituir, texto)


def texto_para_tts(frase: str) -> str:
    """Aplica config/pronunciacion.yaml y pasa las cifras a palabras. Solo para el texto del TTS.

    El diccionario va primero para que un término con cifras (p. ej. "GPT-4") use su entrada.
    Con `erre_inicial: true` en config/voz.yaml, además duplica la r inicial (ver erre_inicial).
    """
    if config.cargar("voz").get("erre_inicial"):
        frase = erre_inicial(frase)
    compilado = _patron_pronunciacion()
    if compilado is not None:
        patron, dic = compilado
        frase = patron.sub(lambda m: dic[m.group(0).lower()], frase)
    return numeros_a_palabras(frase)


def terminos_pronunciacion() -> set[str]:
    """Términos del diccionario y sus escrituras fonéticas, para excluirlos del WER
    (Whisper puede transcribir cualquiera de las dos formas)."""
    dic = config.cargar("pronunciacion")
    return {t.lower() for par in dic.items() for t in par}


# --- Números -----------------------------------------------------------------
# El TTS lee mal las cifras ("0,001") y Whisper las transcribe a veces en cifras y a veces
# en letras. Se expanden a palabras antes del TTS y en ambos lados del WER.

# 150 528 / 150.528 (miles) · 0,001 / 2.5 (decimales) · 224×224 · 12 %
_MILES = re.compile(r"(?<![\d,.])\d{1,3}(?:[ .]\d{3})+(?![\d,]|\.\d)")
_NUMERO = re.compile(r"(?<![\w,.])(\d+)(?:[,.](\d+))?(?!\w)")


def _numero_a_palabras(entero: str, decimales: str | None) -> str:
    from num2words import num2words

    texto = num2words(int(entero), lang="es")
    if decimales:
        ceros = len(decimales) - len(decimales.lstrip("0"))
        resto = decimales.lstrip("0")
        partes = ["cero"] * ceros + ([num2words(int(resto), lang="es")] if resto else [])
        texto += " coma " + " ".join(partes)
    return texto


def numeros_a_palabras(texto: str) -> str:
    """'0,001' → 'cero coma cero cero uno'; '224×224' → 'doscientos veinticuatro por …'; '12 %' → '… por ciento'."""
    texto = _MILES.sub(lambda m: re.sub(r"[ .]", "", m.group(0)), texto)
    texto = re.sub(r"(?<=\d)\s*[×x]\s*(?=\d)", " por ", texto)
    texto = re.sub(r"(?<=\d)\s*%", " por ciento", texto)
    return _NUMERO.sub(lambda m: _numero_a_palabras(m.group(1), m.group(2)), texto)


def normalizar_para_wer(texto: str, excluir: set[str] = frozenset()) -> str:
    """Minúsculas, cifras en palabras, sin tildes ni puntuación y sin los términos de `excluir`.

    Los términos se quitan como secuencia completa ("ge ene ene"), nunca palabra a palabra,
    para no descartar palabras comunes como "te" o "u".
    """
    texto = unicodedata.normalize("NFD", numeros_a_palabras(texto).lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = " " + " ".join(re.sub(r"[^\w\s]", " ", texto).split()) + " "
    for termino in sorted((normalizar_para_wer(t) for t in excluir), key=len, reverse=True):
        if termino:
            texto = texto.replace(f" {termino} ", " ")
    return " ".join(texto.split())
