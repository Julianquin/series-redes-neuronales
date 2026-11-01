"""Voz del episodio + tiempos por palabra con faster-whisper.

Motor por defecto: **Qwen3-TTS**, el mismo narrador que la serie larga (config/voz.yaml). Se eligió
el 2026-09-20 tras compararlo con Chatterbox sobre las mismas frases: Chatterbox metía acento inglés
en «red», vocalizaciones sueltas y colas que había que recortar. Compartir narrador entre los dos
canales es lo que hace reconocible al autor cuando cambia el formato.

    .venv-qwen/bin/python explainer/tts.py explainer/guion_ep01_mlp.yaml [--solo s03]
    .venv/bin/python      explainer/tts.py ... --motor chatterbox      (el de antes, comparado)

Cada motor vive en su intérprete: Qwen en .venv-qwen, Chatterbox en .venv. La referencia de voz y la
velocidad salen de config/voz.yaml de la serie salvo que se pasen a mano.

Salida: voz/<guion>/<id>.wav (mono, 44,1 kHz, -16 LUFS) y voz/<guion>/timings.json, que explainer.py
usa automáticamente. La velocidad se aplica ANTES de medir: lo que se mide es lo que se publica."""
import argparse, difflib, json, os, re, subprocess, unicodedata
import yaml

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # Xet se colgó dos veces en esta máquina
os.environ.pop("HF_HUB_OFFLINE", None)             # el cargador de Qwen consulta el Hub

HERE = os.path.dirname(os.path.abspath(__file__))
def norm(w): return unicodedata.normalize("NFC", re.sub(r"[^\wáéíóúüñ]", "", w.lower()))

def raiz_serie():
    """La carpeta de la serie larga, de donde salen el narrador y su configuración."""
    d = HERE
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "config", "voz.yaml")): return d
        d = os.path.dirname(d)
    return None

def pronounce(text, table):
    for k, v in (table or {}).items(): text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text

def interpolar(t, dur):
    """Rellena los huecos de una lista de tiempos repartiéndolos entre los vecinos conocidos."""
    known = [(i, x) for i, x in enumerate(t) if x is not None]
    if not known: return [dur * (i + .5) / len(t) for i in range(len(t))]
    for i in range(len(t)):
        if t[i] is None:
            prev = max((k for k in known if k[0] < i), default=(-1, 0.0))
            nxt = min((k for k in known if k[0] > i), default=(len(t), dur))
            t[i] = prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / (nxt[0] - prev[0])
    return t


def mapear(orig, pron, t_pron, dur):
    """Lleva los tiempos de lo pronunciado a las palabras del guion.

    El diccionario de pronunciación puede cambiar una palabra por varias («LSTM» → «ele ese te
    eme») o cubrir varias a la vez («deep learning»), así que las posiciones no coinciden. Se
    emparejan con difflib y, dentro de un tramo sustituido, se reparten proporcionalmente. Los
    subtítulos siguen mostrando el texto del guion; solo los tiempos vienen de lo que se dijo.
    """
    t = [None] * len(orig)
    a = [norm(w) for w in orig]; b = [norm(w) for w in pron]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1): t[i1 + k] = t_pron[j1 + k]
        elif j2 > j1 and i2 > i1:
            for k in range(i2 - i1):
                t[i1 + k] = t_pron[min(j2 - 1, j1 + k * (j2 - j1) // (i2 - i1))]
    return interpolar(t, dur)


def align(guion_words, whisper_words, dur):
    """Asigna a cada palabra del guion el instante de su palabra en Whisper; las no emparejadas se interpolan."""
    a = [norm(w) for w in guion_words]; b = [norm(w["w"]) for w in whisper_words]; t = [None] * len(a)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size): t[blk.a + k] = whisper_words[blk.b + k]["t"]
    return interpolar(t, dur)

def cargar_motor(nombre, voz_cfg):
    """Devuelve sintetiza(texto) -> (wav numpy mono, sample rate). Un solo modelo en la GPU."""
    if nombre == "qwen":
        import torch
        from qwen_tts import Qwen3TTSModel
        modelo = Qwen3TTSModel.from_pretrained(voz_cfg["qwen"]["modelo"], device_map="cuda:0", dtype=torch.bfloat16)
        ref, ref_texto = voz_cfg["_ref"], voz_cfg["_ref_texto"]
        idioma = voz_cfg["qwen"]["idioma"]
        def sintetiza(texto):
            wavs, sr = modelo.generate_voice_clone(text=texto, language=idioma, ref_audio=ref, ref_text=ref_texto)
            return wavs[0], sr
        return sintetiza, modelo
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS   # el motor anterior, para comparar
    p = (voz_cfg.get("chatterbox") or {}).get("parametros", {})
    modelo = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    def sintetiza(texto):
        wav = modelo.generate(texto, language_id="es", audio_prompt_path=voz_cfg["_ref"],
                              exaggeration=p.get("exaggeration", 0.5), cfg_weight=p.get("cfg_weight", 0.5))
        return wav.squeeze(0).detach().cpu().numpy(), modelo.sr
    return sintetiza, modelo

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("guion")
    ap.add_argument("--motor", default="qwen", choices=("qwen", "chatterbox"))
    ap.add_argument("--ref", default=None, help="voz de referencia (por defecto, la de config/voz.yaml)")
    ap.add_argument("--ref-texto", default=None, help="su transcripción; Qwen la necesita")
    ap.add_argument("--velocidad", type=float, default=None, help="1.0 = sin tocar (por defecto, la de config/voz.yaml)")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--solo", default=None, help="regenerar solo este segmento")
    ap.add_argument("--whisper", default="large-v3")
    args = ap.parse_args()

    G = yaml.safe_load(open(args.guion))
    raiz = raiz_serie()
    voz_cfg = yaml.safe_load(open(os.path.join(raiz, "config", "voz.yaml"))) if raiz else {}
    voz_cfg.setdefault("qwen", {"modelo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "idioma": "Spanish"})
    voz_cfg["_ref"] = args.ref or (os.path.join(raiz, voz_cfg["referencia"]) if raiz and "referencia" in voz_cfg else None)
    voz_cfg["_ref_texto"] = args.ref_texto or voz_cfg.get("texto_referencia", "")
    vel = args.velocidad if args.velocidad is not None else float(voz_cfg.get("velocidad", 1.0))
    if not voz_cfg["_ref"] or not os.path.exists(voz_cfg["_ref"]):
        raise SystemExit(f"No encuentro la voz de referencia ({voz_cfg['_ref']}): pásala con --ref")

    VOZ = os.path.join(HERE, "voz", os.path.splitext(os.path.basename(args.guion))[0]); os.makedirs(VOZ, exist_ok=True)
    import numpy as np, soundfile as sf, torch                 # soundfile, no torchaudio: en 2.11 save exige TorchCodec
    segs = [s for s in G["segmentos"] if args.solo in (None, s["id"])]
    print(f"motor {args.motor} · velocidad {vel} · {len(segs)} segmento(s) · referencia {os.path.basename(voz_cfg['_ref'])}")

    dicho = {}                                                  # lo que de verdad se le dio al motor
    sintetiza, modelo = cargar_motor(args.motor, voz_cfg)
    for s in segs:
        torch.manual_seed(args.seed)                           # misma semilla => misma voz en todos los episodios
        dicho[s["id"]] = pronounce(s["texto"], G.get("pronunciacion"))
        wav, sr = sintetiza(dicho[s["id"]])
        raw = os.path.join(VOZ, f"{s['id']}_raw.wav"); sf.write(raw, np.asarray(wav, dtype="float32"), sr)
        out = os.path.join(VOZ, f"{s['id']}.wav")
        filtro = (f"atempo={vel:.4f}," if abs(vel - 1.0) > 1e-3 else "") + "loudnorm=I=-16:TP=-1.5:LRA=11"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af", filtro,
                        "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", out], check=True)
        os.remove(raw); print("voz:", out)
    del modelo, sintetiza; torch.cuda.empty_cache()             # un modelo cada vez en la GPU

    tp = os.path.join(VOZ, "timings.json"); timings = json.load(open(tp)) if os.path.exists(tp) else {}
    from faster_whisper import WhisperModel
    wm = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    for s in segs:
        p = os.path.join(VOZ, f"{s['id']}.wav")
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
        res, _ = wm.transcribe(p, language="es", word_timestamps=True)
        ww = [{"w": w.word.strip(), "t": w.start} for seg in res for w in seg.words]
        # Se compara y se alinea contra lo que se le dio al motor, no contra el texto del guion:
        # si no, cada entrada del diccionario cuenta como fallo (NumPy → nám-pai se oye «nampy»).
        # Los tiempos siguen siendo por posición, así que los subtítulos muestran el texto original.
        ref = dicho.get(s["id"], s["texto"])
        t_pron = align(ref.split(), ww, dur)
        palabras = t_pron if ref == s["texto"] else mapear(s["texto"].split(), ref.split(), t_pron, dur)
        timings[s["id"]] = {"dur": dur, "words": palabras}
        heard = " ".join(x["w"] for x in ww); ratio = difflib.SequenceMatcher(None, norm(ref), norm(heard)).ratio()
        flag = "  ⚠ revisar pronunciación" if ratio < 0.85 else ""
        print(f"{s['id']}: {dur:5.1f}s · {len(s['texto'].split()) / dur:.2f} palabras/s · parecido {ratio:.2f}{flag}")
    json.dump(timings, open(tp, "w"), ensure_ascii=False, indent=1); print("tiempos:", tp)

if __name__ == "__main__":
    main()
