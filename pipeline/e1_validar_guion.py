"""Etapa 1 — Validar guion.yaml (§9): estructura, longitudes, marcas y componentes.

Los errores detienen el pipeline; los avisos se muestran pero no bloquean.
"""
import re
import sys

from lib import config
from lib.texto import dividir_frases
from pipeline import comun

CAMPOS_EPISODIO = ("episodio", "slug", "titulo", "duracion_objetivo_min", "segmentos")
CAMPOS_SEGMENTO = ("id", "bloque", "escena", "narracion", "visual")


def componentes_existentes() -> set[str]:
    clases = set()
    for py in (config.RAIZ / "lib" / "componentes").glob("*.py"):
        clases |= set(re.findall(r"^class\s+(\w+)", py.read_text(encoding="utf-8"), re.M))
    return clases


def validar_musica(guion: dict, ids: set[str]) -> list[str]:
    """El campo `musica`: cadena (una pista en bucle) o lista de tramos con pista/desde/entrada."""
    spec = guion.get("musica")
    if not spec:
        return []
    tramos = [{"pista": spec}] if isinstance(spec, str) else spec
    errores = []
    for t in tramos:
        if not isinstance(t, dict) or "pista" not in t:
            errores.append(f"musica: tramo sin campo 'pista': {t!r}")
            continue
        if not (config.RAIZ / t["pista"]).exists():
            errores.append(f"musica: no existe {t['pista']}")
        for campo in ("desde", "entrada"):
            if t.get(campo) and t[campo] not in ids:
                errores.append(f"musica: '{campo}: {t[campo]}' no es un segmento del guion")
        if t.get("transicion") not in (None, "auto") and not isinstance(t.get("transicion"), (int, float)):
            errores.append(f"musica: 'transicion' debe ser 'auto' o un número de segundos, no {t['transicion']!r}")
        if t.get("transicion") and not t.get("entrada"):
            errores.append("musica: 'transicion' necesita 'entrada' (el segmento donde debe caer)")
    return errores


def validar(guion: dict) -> tuple[list[str], list[str]]:
    errores, avisos = [], []
    reglas = config.cargar("pipeline")["guion"]

    for campo in CAMPOS_EPISODIO:
        if campo not in guion:
            errores.append(f"falta el campo de episodio '{campo}'")
    if errores:
        return errores, avisos

    disponibles = componentes_existentes()
    ids, escenas, nuevos = set(), set(), set()
    caracteres, pausas = 0, 0.0

    for n, seg in enumerate(guion["segmentos"], 1):
        sid = seg.get("id", f"#{n}")
        faltan = [c for c in CAMPOS_SEGMENTO if not seg.get(c)]
        if faltan:
            errores.append(f"{sid}: faltan campos {faltan}")
            continue
        if sid in ids:
            errores.append(f"{sid}: id duplicado")
        if seg["escena"] in escenas:
            errores.append(f"{sid}: escena '{seg['escena']}' duplicada")
        ids.add(sid)
        escenas.add(seg["escena"])

        if seg["bloque"] not in comun.BLOQUES:
            errores.append(f"{sid}: bloque '{seg['bloque']}' no válido (usa: {', '.join(comun.BLOQUES)})")

        frases = dividir_frases(seg["narracion"])
        for f in frases:
            caracteres += len(f.texto)
            pausas += f.pausa_extra
            if f.palabras > reglas["max_palabras_frase"]:
                avisos.append(f"{sid}: frase de {f.palabras} palabras (> {reglas['max_palabras_frase']}): «{f.texto}»")

        narracion = " ".join(f.texto for f in frases)
        eventos, posicion_anterior, evento_anterior = set(), -1, None
        for marca in seg.get("marcas") or []:
            palabra, evento = marca.get("palabra"), marca.get("evento")
            if not palabra or not evento:
                errores.append(f"{sid}: marca incompleta {marca}")
                continue
            apariciones = [m.start() for m in
                           re.finditer(rf"(?<!\w){re.escape(palabra)}(?!\w)", narracion, re.IGNORECASE)]
            if not apariciones:
                errores.append(f"{sid}: la marca «{palabra}» no aparece en la narración")
                continue
            n = int(marca.get("ocurrencia", 1))
            if n > len(apariciones):
                errores.append(f"{sid}: «{palabra}» aparece {len(apariciones)} vez/veces, "
                               f"no {n} (revisa 'ocurrencia')")
                continue
            if len(apariciones) > 1 and "ocurrencia" not in marca:
                avisos.append(f"{sid}: «{palabra}» aparece {len(apariciones)} veces; se usa la primera "
                              f"(añade 'ocurrencia: 2' si querías otra)")
            # El orden de las marcas debe ser el de la narración: si no, las animaciones llegan tarde.
            if apariciones[n - 1] < posicion_anterior:
                errores.append(f"{sid}: '{evento}' («{palabra}») va antes que '{evento_anterior}' en la "
                               f"narración; ordena las marcas como se dicen")
            posicion_anterior, evento_anterior = apariciones[n - 1], evento
            if evento in eventos:
                errores.append(f"{sid}: evento '{evento}' duplicado")
            eventos.add(evento)

        nuevos |= set(seg.get("componentes") or []) - disponibles

    if nuevos:
        avisos.append(f"componentes nuevos que Claude Code debe crear en lib/componentes/: {', '.join(sorted(nuevos))}")

    estimado_min = (caracteres / reglas["caracteres_por_segundo"] + pausas) / 60
    objetivo = guion["duracion_objetivo_min"]
    desvio = (estimado_min - objetivo) / objetivo
    resumen = f"duración estimada {estimado_min:.1f} min frente a {objetivo} min objetivo ({desvio:+.0%})"
    if abs(desvio) > reglas["tolerancia_duracion"]:
        avisos.append(resumen)
    else:
        print(f"[OK] {resumen}")

    errores += validar_musica(guion, ids)

    bloques = [s.get("bloque") for s in guion["segmentos"]]
    if bloques and bloques[0] != "gancho":
        avisos.append("el primer segmento no es un 'gancho'")
    return errores, avisos


def main() -> int:
    args = comun.argumentos(__doc__)
    dir_ep = config.dir_episodio(args.ep)
    guion = comun.cargar_guion(dir_ep)
    errores, avisos = validar(guion)
    n = len(guion.get("segmentos") or [])
    for a in avisos:
        print(f"[AVISO] {a}")
    for e in errores:
        print(f"[FALLO] {e}")
    print(f"{dir_ep.name}: {n} segmentos, {len(errores)} errores, {len(avisos)} avisos")
    return comun.FALLO if errores else comun.OK


if __name__ == "__main__":
    sys.exit(main())
