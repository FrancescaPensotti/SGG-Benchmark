#!/usr/bin/env python3
"""
Test offline dello Stadio D semantico, senza ROS ne' Gemini: il valutatore
LLM e' sostituito da una funzione finta che conta le chiamate.
Importa il modulo vero (demo/functional_relations.py).

Uso: python3 demo/test_functional_relations.py   (oppure pytest)
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functional_relations import FunctionalRelationTable, choose_next  # noqa: E402

FAKE = {
    'bottle': {'cup': 0.9, 'glass': 0.85, 'banana': 0.1, 'orange': 0.1},
    'banana': {'orange': 0.6, 'bowl': 0.7, 'bottle': 0.1, 'cup': 0.0},
}


class FakeScorer:
    def __init__(self):
        self.calls = []

    def __call__(self, grasped, candidates):
        self.calls.append((grasped, list(candidates)))
        return {c: {'score': FAKE.get(grasped, {}).get(c, 0.0), 'reason': f'{grasped}->{c}'}
                for c in candidates}


def tmp_path():
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.remove(path)
    return path


def test_sceglie_il_punteggio_piu_alto():
    t = FunctionalRelationTable(tmp_path(), FakeScorer())
    s = t.scores('bottle', ['banana', 'cup', 'orange'])
    assert choose_next(s, 0.5)[0] == 'cup'


def test_sotto_soglia_nessun_suggerimento():
    t = FunctionalRelationTable(tmp_path(), FakeScorer())
    s = t.scores('bottle', ['banana', 'orange'])
    assert choose_next(s, 0.5) is None


def test_oggetto_afferrato_escluso():
    scorer = FakeScorer()
    t = FunctionalRelationTable(tmp_path(), scorer)
    s = t.scores('bottle', ['bottle', 'cup'])
    assert 'bottle' not in s and scorer.calls == [('bottle', ['cup'])]


def test_una_richiesta_per_oggetto_afferrato_poi_cache():
    scorer = FakeScorer()
    t = FunctionalRelationTable(tmp_path(), scorer)
    t.scores('bottle', ['cup', 'glass', 'banana'])
    t.scores('bottle', ['cup', 'glass', 'banana'])
    assert len(scorer.calls) == 1, scorer.calls


def test_chiede_solo_le_coppie_mancanti():
    scorer = FakeScorer()
    t = FunctionalRelationTable(tmp_path(), scorer)
    t.scores('bottle', ['cup'])
    t.scores('bottle', ['cup', 'glass'])
    assert scorer.calls == [('bottle', ['cup']), ('bottle', ['glass'])]


def test_tabella_salvata_e_riletta_senza_rete():
    path = tmp_path()
    FunctionalRelationTable(path, FakeScorer()).scores('bottle', ['cup', 'banana'])
    scorer = FakeScorer()
    t = FunctionalRelationTable(path, scorer)
    s = t.scores('bottle', ['cup', 'banana'])
    assert scorer.calls == [] and choose_next(s)[0] == 'cup'
    with open(path) as f:
        assert json.load(f)['pairs']['bottle']['cup']['score'] == 0.9


def test_stesso_risultato_indipendente_dalla_scena():
    # Stessa coppia valutata in scene diverse: il punteggio e' quello salvato.
    t = FunctionalRelationTable(tmp_path(), FakeScorer())
    a = t.scores('bottle', ['cup', 'banana'])['cup']
    b = t.scores('bottle', ['cup', 'orange', 'glass'])['cup']
    assert a == b


def test_senza_valutatore_usa_solo_la_tabella():
    t = FunctionalRelationTable(tmp_path(), None)
    assert t.scores('bottle', ['cup']) == {}
    t.pairs = {'bottle': {'cup': {'score': 0.8, 'reason': 'a mano'}}}
    assert choose_next(t.scores('bottle', ['cup', 'glass']))[0] == 'cup'


def test_valutatore_che_fallisce_non_scrive_nulla():
    path = tmp_path()
    t = FunctionalRelationTable(path, lambda g, c: None)
    assert t.scores('bottle', ['cup']) == {} and not os.path.exists(path)


def test_risposta_sporca_ripulita():
    t = FunctionalRelationTable(tmp_path(), lambda g, c: {
        'cup': {'score': 1.7, 'reason': 'ok'}, 'glass': {'score': 'boh'}})
    s = t.scores('bottle', ['cup', 'glass', 'banana'])
    assert s['cup']['score'] == 1.0 and s['glass']['score'] == 0.0 and 'banana' not in s


def test_parita_decisa_in_ordine_alfabetico():
    s = {'glass': {'score': 0.9, 'reason': ''}, 'cup': {'score': 0.9, 'reason': ''}}
    assert choose_next(s)[0] == 'cup'


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FALLITO {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test superati")
    sys.exit(1 if failed else 0)
