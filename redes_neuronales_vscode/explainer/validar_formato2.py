"""Comprueba un guion de formato 2 contra su módulo sin renderizar ni cargar los datos entrenados.

    python3 validar_formato2.py                 # todos los guiones con formato: 2
    python3 validar_formato2.py guion_ep02_cnn.yaml

Pilla los fallos que no se ven hasta que el render falla o, peor, sale mal: un óvalo que no está en
el código, una línea fuera del bloque, una fórmula o una curva que el módulo no define, o un cue
citado en el piso 2 que no existe en la narración.
"""
import glob, os, re, sys, unicodedata
import yaml

norm = lambda w: unicodedata.normalize('NFC', re.sub(r'[^\w\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]', '', w.lower()))

TIPOS = {"codigo", "banner", "formula", "lista", "ficha", "curva"}
COLORES = {"verde", "oro", "rojo", "azul", "gris"}
VIS = 10          # líneas visibles del bloque de código (explainer.low_code)


def extras_del_modulo(py):
    """Lee CODE y los nombres de fórmulas y curvas sin importar el módulo (necesitaría los .pkl)."""
    s = open(py, encoding="utf-8").read()
    m = re.search(r'^CODE = ("""|\'\'\')(.*?)\1', s, re.S | re.M)
    code = m.group(2).strip("\n").split("\n") if m else []
    bloque = re.search(r"_extras=dict\((.*?)\n\S", s, re.S)
    txt = bloque.group(1) if bloque else ""
    saca = lambda k: set(re.findall(r'"([^"]+)":', re.search(k + r"=\{(.*?)\}", txt, re.S).group(1))) if re.search(k + r"=\{", txt) else set()
    return code, saca("formulas"), saca("curvas" if "curvas={" in txt else "curves"), ("_extras=dict" in s)


def dibuja_la_escena(py):
    """Qué pinta ya cada escena del piso 1: fórmula, gráfica o bloque de código.

    El piso 2 no debe repetirlo. La regla, sacada de migrar el episodio 02: si la escena ya dibuja
    una fórmula, abajo va el código; si ya dibuja una gráfica, la gráfica se baja al piso 2 y se
    quita de la escena; si la escena es la del código, abajo va el resumen en lista.
    """
    s = open(py, encoding="utf-8").read()
    fuera = {}
    for chunk in re.split(r"\ndef ", s):
        nombre = chunk.split("(")[0].strip()
        cuerpo = chunk[:chunk.index("\n\n")] if "\n\n" in chunk else chunk
        if not nombre.isidentifier(): continue
        marcas = set()
        if "chart(" in cuerpo: marcas.add("curva")
        if ".draw(" in cuerpo: marcas.add("formula")
        if "CodeBlock" in cuerpo: marcas.add("codigo")
        if marcas: fuera[nombre.rstrip("_")] = marcas
    return fuera


def revisar(ruta):
    G = yaml.safe_load(open(ruta, encoding="utf-8"))
    if G.get("formato", 1) < 2: return []
    base = os.path.join(os.path.dirname(os.path.abspath(ruta)), "")
    py = sorted(glob.glob(f"{base}ep{G['episodio']:02d}_*.py"))
    fallos, avisos = [], []
    def mal(donde, txt): fallos.append(f"{os.path.basename(ruta)} · {donde}: {txt}")
    def ojo(donde, txt): avisos.append(f"{os.path.basename(ruta)} · {donde}: {txt}")

    for k in ("lema", "promesas", "ficha", "arquitectura"):
        if not G.get(k): mal("cabecera", f"falta «{k}», que el formato 2 dibuja")
    if len(G.get("promesas", [])) > 3:
        mal("cabecera", f"{len(G['promesas'])} promesas: el banner solo tiene sitio cómodo para 3")
    for i, fila in enumerate(G.get("ficha", [])):
        if len(fila) < 3 or fila[2] not in COLORES:
            mal("ficha", f"fila {i + 1}: color {fila[2] if len(fila) > 2 else '(ninguno)'} no está en {sorted(COLORES)}")

    if not py:
        mal("módulo", f"no encuentro ep{G['episodio']:02d}_*.py"); return fallos, avisos
    code, formulas, curvas, tiene_extras = extras_del_modulo(py[0])
    ya_dibuja = dibuja_la_escena(py[0])
    if not tiene_extras: mal(os.path.basename(py[0]), "SCENES no tiene _extras: el piso 2 no podrá dibujar nada")

    for seg in G["segmentos"]:
        # Regla del README: un cue en las últimas palabras dispara su animación cuando el segmento
        # ya se acaba, así que apenas se ve. Se mide en palabras que quedan por decir.
        pal = seg["texto"].split()
        for nombre, palabra in (seg.get("cues") or {}).items():
            pos = next((i for i, w in enumerate(pal) if norm(w) == norm(palabra)), None)
            if pos is None: mal(f"{seg['id']}/{nombre}", f"la palabra «{palabra}» no está en la narración")
            elif len(pal) - 1 - pos <= 2:
                ojo(f"{seg['id']}/{nombre}", f"«{palabra}» cae a {len(pal) - 1 - pos} palabra(s) del final: la animación casi no se ve")

        p2 = seg.get("piso2")
        if not p2: continue
        cues = set((seg.get("cues") or {}))
        for spec in (p2 if isinstance(p2, list) else [p2]):
            d = f"{seg['id']}/{spec.get('tipo')}"
            if spec.get("tipo") not in TIPOS: mal(d, f"tipo desconocido (válidos: {sorted(TIPOS)})"); continue
            repe = ya_dibuja.get(str(seg.get("escena", "")).rstrip("_"), set())
            if spec["tipo"] in repe:
                ojo(d, f"la escena «{seg['escena']}» ya dibuja {spec['tipo']} en el piso 1: "
                       "se vería dos veces (baja eso al piso 2 y quítalo de la escena, o cambia de tipo)")
            if spec.get("cue") and spec["cue"] not in cues: mal(d, f"cue «{spec['cue']}» no está en los cues del segmento")
            if spec["tipo"] == "codigo":
                ln = spec.get("lineas", [])
                fuera = [i for i in ln if not 0 <= i < len(code)]
                if fuera: mal(d, f"líneas fuera del bloque ({len(code)} líneas): {fuera}"); continue
                if ln and max(ln) - min(ln) >= VIS: mal(d, f"líneas {min(ln)}–{max(ln)}: no caben en las {VIS} visibles")
                ov = spec.get("ovalo")
                if ov and not any(ov in code[i] for i in ln):
                    donde = [i for i, l in enumerate(code) if ov in l]
                    mal(d, f"el óvalo «{ov}» no está en las líneas activas" + (f" (sí en {donde})" if donde else " ni en ninguna"))
                if spec.get("ovalo_cue") and spec["ovalo_cue"] not in cues: mal(d, f"ovalo_cue «{spec['ovalo_cue']}» no existe")
            elif spec["tipo"] == "formula" and spec.get("nombre") not in formulas:
                mal(d, f"fórmula «{spec.get('nombre')}» no está en _extras (hay: {sorted(formulas) or 'ninguna'})")
            elif spec["tipo"] == "curva" and spec.get("nombre") not in curvas:
                mal(d, f"curva «{spec.get('nombre')}» no está en _extras (hay: {sorted(curvas) or 'ninguna'})")
    return fallos, avisos


def main():
    rutas = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "guion_ep*.yaml")))
    total = 0
    for r in rutas:
        G = yaml.safe_load(open(r, encoding="utf-8"))
        if G.get("formato", 1) < 2: print(f"  {os.path.basename(r):28} formato 1, se salta"); continue
        f, a = revisar(r)
        total += len(f)
        estado = "OK" if not f else f"{len(f)} FALLO(S)"
        print(f"  {os.path.basename(r):28} {estado}" + (f" · {len(a)} aviso(s)" if a else ""))
        for x in f: print(f"      FALLO  {x}")
        for x in a: print(f"      aviso  {x}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
