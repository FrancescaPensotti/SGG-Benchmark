# demo/gemini_retrieval.py
# ──────────────────────────────────────────────────────────────
# Retrieval dei target in linguaggio naturale tramite Gemini.
#
# Idea: l'operatore non digita la label esatta della classe
# (es. "glass"), ma descrive il target a parole sue
# (es. "la cosa con cui bevo"). Gemini riceve il grafo della
# scena corrente + la descrizione e restituisce solo le label
# degli oggetti realmente presenti nella scena che corrispondono.
#
# Due funzioni:
#   - scene_graph_to_json(scene_graph)  → serializza il grafo
#   - resolve_targets(descrizione, scene_json) → lista di label
#
# Decoupling: questo file riceve solo lo scene_graph come argomento, così resta autonomo
# e testabile da solo.
# ──────────────────────────────────────────────────────────────

import os
import json
import time

from google import genai
from google.genai import types

# ── Configurazione ──────────────────────────────────────────
MODEL = "gemini-2.5-flash"   # free tier "gemini-2.5-flash-lite"
FREQ_THRESHOLD = 2           # stesso filtro frequenza usato nel nodo ROS2

# Client creato una sola volta (lazy) e riusato.
_client = None


def get_client():
    """
    Crea (una volta) il client Gemini leggendo la chiave dalla
    variabile d'ambiente GEMINI_API_KEY.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY non impostata. Aggiungi in ~/.bashrc:\n"
                '  export GEMINI_API_KEY="la_tua_chiave"\n'
                "poi: source ~/.bashrc"
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ── 1. Serializzazione del grafo semantico in JSON ──────────
def scene_graph_to_json(scene_graph, freq_threshold=FREQ_THRESHOLD):
    """
    Converte lo scene_graph corrente in un JSON compatto da dare
    a Gemini come contesto. Include solo gli oggetti "stabili"
    (visti almeno freq_threshold volte).

    Struttura prodotta:
        {
          "objects": [
            {"id": 2, "label": "glass", "count": 17,
             "position_pixel": [261, 321]},
            ...
          ],
          "relations": [
            {"subject": "hand", "relation": "holding", "object": "glass"},
            ...
          ]
        }

    Le relazioni servono a disambiguare descrizioni tipo
    "la bottiglia che la mano sta tenendo".
    """
    objects = []
    for i, node in enumerate(scene_graph):
        if node['count'] < freq_threshold:
            continue
        objects.append({
            "id": i,
            "label": node['label'],
            "count": node['count'],
            "position_pixel": list(node.get('position', (0, 0))),
        })

    # Le relazioni puntano all'uid del nodo (vedi sgg_ros_node.py); i grafi
    # senza uid (es. il grafo finto qui sotto) usano l'indice come uid.
    nodes_by_uid = {n.get('uid', i): n for i, n in enumerate(scene_graph)}
    relations = []
    for node in scene_graph:
        if node['count'] < freq_threshold:
            continue
        for (pred, obj_uid), count in list(node['relazioni'].items()):
            if count < freq_threshold:
                continue
            obj = nodes_by_uid.get(obj_uid)
            if obj is None:
                continue
            relations.append({
                "subject": node['label'],
                "relation": pred,
                "object": obj['label'],
            })

    return {"objects": objects, "relations": relations}


# ── 2. Retrieval dei target via Gemini ──────────────────────
def resolve_targets(description, scene_json, model=MODEL):
    """
    Dato il JSON della scena e una descrizione in linguaggio
    naturale, chiede a Gemini quali oggetti della scena
    corrispondono.

    Ritorna una lista di label (stringhe), ordinata per
    rilevanza/ordine d'azione. Lista vuota se niente corrisponde.

    Vincolo importante: Gemini DEVE scegliere solo tra le label
    realmente presenti nella scena — non deve inventare oggetti.
    """
    client = get_client()

    # Insieme delle label disponibili (per vincolare l'output).
    labels_present = sorted({o['label'] for o in scene_json['objects']})

    if not labels_present:
        return []

    prompt = f"""Sei l'assistente semantico di un robot in teleoperazione.
Hai a disposizione il grafo della scena osservata dalla telecamera.

SCENA (JSON):
{json.dumps(scene_json, ensure_ascii=False, indent=2)}

LABEL DISPONIBILI NELLA SCENA:
{labels_present}

L'operatore descrive l'oggetto (o gli oggetti) che vuole raggiungere
con parole sue, non con la label esatta:
"{description}"

Compito: individua quali oggetti della scena corrispondono alla
descrizione. Regole:
- Scegli ESCLUSIVAMENTE tra le LABEL DISPONIBILI elencate sopra.
- Non inventare oggetti che non sono nella lista.
- Se la descrizione indica piu' target in sequenza (es. "prendi prima
  la bottiglia, poi il telefono"), restituiscili nell'ordine richiesto.
- Se nessun oggetto corrisponde, restituisci una lista vuota.

Rispondi SOLO con un array JSON di stringhe, senza testo aggiuntivo.
Esempio di output valido: ["glass"]  oppure  ["bottle", "phone"]  oppure  []
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"[gemini_retrieval] Tentativo {attempt+1} fallito: {e}. Riprovo tra 3s...")
                time.sleep(3)
            else:
                print(f"[gemini_retrieval] Tutti i tentativi falliti: {e}")
                return []

    # Parsing robusto della risposta.
    raw = (response.text or "").strip()
    # Rimuove eventuali fence ```json ... ``` se presenti.
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[gemini_retrieval] Risposta non parsabile: {raw!r}")
        return []

    if not isinstance(result, list):
        return []

    # Filtro finale di sicurezza: tieni solo label realmente presenti.
    valid = [lbl for lbl in result if lbl in labels_present]
    return valid


# ── 3. Giudice di plausibilità funzionale (avanzamento post-grasp) ──
def judge_functional_relation(label_a, predicate, label_b, model=MODEL):
    """
    Chiamata da advance_to_next_object() dopo che un grasp è confermato:
    valuta se la relazione rilevata tra l'oggetto appena graspato (label_a)
    e un candidato successivo (label_b) riflette un legame funzionale reale
    (non solo prossimità/impilamento casuale), ancorando il giudizio alla
    relazione VG150 effettivamente osservata (predicate) invece di chiedere
    "sono mai correlati in astratto" -- domanda troppo vaga, un LLM tende a
    rispondere sempre di sì.

    Ritorna un dict {"plausible": bool, "score": float, "reason": str}, o
    None se Gemini non è raggiungibile o la risposta non è parsabile (stesso
    fallback sicuro di resolve_targets) -- il chiamante decide cosa fare in
    quel caso.
    """
    client = get_client()

    prompt = f"""Sei l'assistente semantico di un robot che, dopo aver afferrato un oggetto,
decide se muoversi autonomamente verso un secondo oggetto perché
funzionalmente collegato al primo (es. presa una bottiglia, andare verso
il bicchiere per versare).

Il sistema di percezione ha rilevato questa relazione tra due oggetti nella scena:
"{label_a}" -- {predicate} --> "{label_b}"

Appena afferrato "{label_a}", ha senso che il robot si muova autonomamente
verso "{label_b}" come prossimo passo di un compito reale? Considera:
- Una relazione puramente spaziale/casuale (es. due oggetti solo vicini o
  impilati per caso) NON è funzionale.
- Una relazione che riflette un uso reale insieme (versare, aprire, usare
  uno strumento sull'altro, ecc.) È funzionale.

Rispondi SOLO con un oggetto JSON, senza testo aggiuntivo, in questo formato:
{{"plausible": true/false, "score": 0.0-1.0, "reason": "breve motivazione"}}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as e:
            if attempt < 2:
                print(f"[gemini_retrieval] Tentativo {attempt+1} fallito: {e}. Riprovo tra 3s...")
                time.sleep(3)
            else:
                print(f"[gemini_retrieval] Tutti i tentativi falliti: {e}")
                return None

    raw = (response.text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[gemini_retrieval] Risposta non parsabile: {raw!r}")
        return None

    if not isinstance(result, dict) or "plausible" not in result:
        print(f"[gemini_retrieval] Risposta in formato inatteso: {result!r}")
        return None

    return {
        "plausible": bool(result.get("plausible")),
        "score": float(result.get("score", 0.0)),
        "reason": str(result.get("reason", "")),
    }


# ── Test standalone (senza ROS2 / senza telecamera) ─────────
if __name__ == "__main__":
    # Mini scene_graph finto per provare la pipeline a tavolino,
    # prima ancora di collegarla al nodo ROS2.
    # Struttura dei nodi identica a quella di sgg_ros_node.py.
    fake_scene_graph = [
        {"label": "woman",  "count": 31, "position": (400, 200),
         "relazioni": {("holding", 2): 12, ("wearing", 3): 9}},
        {"label": "hand",   "count": 24, "position": (350, 300),
         "relazioni": {("holding", 2): 14}},
        {"label": "glass",  "count": 17, "position": (261, 321),
         "relazioni": {}},
        {"label": "shirt",  "count": 20, "position": (410, 260),
         "relazioni": {}},
        {"label": "bottle", "count": 15, "position": (82, 347),
         "relazioni": {}},
    ]

    scene_json = scene_graph_to_json(fake_scene_graph)
    print("Scene JSON:")
    print(json.dumps(scene_json, ensure_ascii=False, indent=2))
    print("\n--- Test retrieval ---")

    for descrizione in [
        "la cosa con cui bevo",
        "il contenitore da cui si versa",
        "prendi prima la bottiglia, poi il bicchiere",
        "il cacciavite",  # non presente -> lista vuota attesa
    ]:
        targets = resolve_targets(descrizione, scene_json)
        print(f'  "{descrizione}"  ->  {targets}')

    print("\n--- Test giudice di plausibilità funzionale ---")
    for label_a, predicate, label_b in [
        ("bottle", "on", "glass"),        # coppia plausibile
        ("book", "on", "orange"),         # coppia-trappola: spaziale, non funzionale
        ("hand", "holding", "glass"),     # plausibile, caso reale del grafo finto
    ]:
        verdict = judge_functional_relation(label_a, predicate, label_b)
        if verdict is None:
            print(f'  "{label_a}" -{predicate}-> "{label_b}"  ->  (Gemini non raggiungibile)')
        else:
            esito = "PLAUSIBILE" if verdict["plausible"] else "NON PLAUSIBILE"
            print(f'  "{label_a}" -{predicate}-> "{label_b}"  ->  {esito} (score={verdict["score"]:.2f}) — {verdict["reason"]}')