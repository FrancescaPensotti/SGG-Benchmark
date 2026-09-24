"""
Relazioni funzionali semantiche per lo Stadio D (24/09/2026): dopo il grasp
di un oggetto, suggerisce il successivo in base ai NOMI degli oggetti in
scena, non alle relazioni spaziali predette da SGG.

Motivo (Metodologia, 24/09): con gli oggetti a ~20 cm l'uno dall'altro SGG
predice quasi solo 'near', che non e' mappato come funzionale, quindi la
regola spaziale (VG_TO_FUNCTIONAL in sgg_ros_node.py) non suggerirebbe quasi
mai niente; e "bottiglia e bicchiere vanno insieme" non dipende da dove sono.

Le valutazioni (punteggio 0-1 + motivazione per ogni coppia ordinata
afferrato -> candidato) vengono da un LLM e si salvano in una tabella JSON su
disco. Una volta riempita (vedi uso da riga di comando sotto), la tabella:
  - non consuma piu' richieste (il piano gratuito di Gemini ne concede 20/giorno);
  - da' gli stessi suggerimenti a tutti i partecipanti della validazione;
  - si puo' leggere e correggere a mano prima delle prove.

Modulo puro, senza ROS: il valutatore LLM si passa come funzione, cosi' i test
girano senza rete (demo/test_functional_relations.py).

Uso da riga di comando (riempie la tabella per gli oggetti della scena di
validazione e la stampa):
    python3 demo/functional_relations.py bottle cup banana orange
    python3 demo/functional_relations.py --mostra
"""

import json
import os

DEFAULT_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'functional_relations_table.json')
# Punteggio minimo perche' un candidato venga suggerito. Valore di partenza,
# da rivedere guardando la tabella riempita.
DEFAULT_MIN_SCORE = 0.5


class FunctionalRelationTable:
    """Tabella afferrato -> candidato -> {'score', 'reason'}, salvata in JSON.

    scorer(grasped_label, candidate_labels) -> {label: {'score', 'reason'}} o
    None se non disponibile: viene chiamato solo per le coppie mancanti, con
    tutte le mancanti dello stesso oggetto afferrato in una sola richiesta.
    """

    def __init__(self, path=DEFAULT_TABLE_PATH, scorer=None):
        self.path = path
        self.scorer = scorer
        self.pairs = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.pairs = json.load(f).get('pairs', {})

    def save(self):
        if not self.path:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'pairs': self.pairs}, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, self.path)

    def lookup(self, grasped, candidate):
        return self.pairs.get(grasped, {}).get(candidate)

    def scores(self, grasped, candidates, allow_query=True):
        """{candidato: {'score', 'reason'}} per i candidati noti (chiedendo al
        valutatore quelli mancanti se allow_query). I candidati che restano
        senza valutazione non compaiono nel risultato."""
        candidates = [c for c in dict.fromkeys(candidates) if c != grasped]
        missing = [c for c in candidates if self.lookup(grasped, c) is None]
        if missing and allow_query and self.scorer is not None:
            answer = self.scorer(grasped, missing)
            if answer:
                row = self.pairs.setdefault(grasped, {})
                for label in missing:
                    if label in answer:
                        row[label] = _clean_entry(answer[label])
                self.save()
        result = {}
        for c in candidates:
            entry = self.lookup(grasped, c)
            if entry is not None:
                result[c] = entry
        return result


def _clean_entry(entry):
    try:
        score = float(entry.get('score', 0.0))
    except (TypeError, ValueError, AttributeError):
        score = 0.0
    reason = str(entry.get('reason', '')) if isinstance(entry, dict) else ''
    return {'score': min(1.0, max(0.0, score)), 'reason': reason}


def choose_next(scores, min_score=DEFAULT_MIN_SCORE):
    """(label, score, reason) del candidato con punteggio piu' alto se
    >= min_score, altrimenti None. A parita' di punteggio vince l'ordine
    alfabetico, per avere lo stesso risultato a ogni prova."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1]['score'], kv[0]))
    if not ranked or ranked[0][1]['score'] < min_score:
        return None
    label, entry = ranked[0]
    return label, entry['score'], entry['reason']


def _print_table(table, labels=None):
    grasped_labels = labels or sorted(table.pairs)
    for g in grasped_labels:
        row = table.pairs.get(g, {})
        cands = [c for c in (labels or sorted(row)) if c != g]
        print(f"\nAfferrato '{g}':")
        for c in sorted(cands, key=lambda c: -(row.get(c) or {}).get('score', -1.0)):
            e = row.get(c)
            if e is None:
                print(f"  {c:>15}:  --   (mancante)")
            else:
                print(f"  {c:>15}: {e['score']:.2f}  {e['reason']}")


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if not args or args == ['--mostra']:
        _print_table(FunctionalRelationTable())
        sys.exit(0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from demo.gemini_retrieval import score_next_objects
    table = FunctionalRelationTable(scorer=score_next_objects)
    for g in args:
        table.scores(g, args)
    _print_table(table, args)
    print(f"\nTabella: {table.path}")
