#!/usr/bin/env python3
"""
Test offline dell'arbitro del target (Stadio C), senza ROS ne' robot.
Importa il modulo vero (demo/target_arbiter.py), non una copia della logica.

Uso: python3 demo/test_target_arbiter.py   (oppure pytest demo/test_target_arbiter.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from target_arbiter import (ArbiterParams, DirectionFilter, TargetArbiter,  # noqa: E402
                            alignment_scores, containment, drop_contained)

EE = (0.0, 0.0, 0.3)          # posizione del robot in base_link [m]
N = ArbiterParams().hysteresis_cycles


def cand(uid, pos, bbox=(0, 0, 100, 100), label='bottle'):
    return {'uid': uid, 'label': label, 'bbox': bbox, 'position_base': pos}


# Due oggetti: A davanti-sinistra, B davanti-destra.
A = cand('A', (0.3, 0.2, 0.0), bbox=(100, 100, 200, 300))
B = cand('B', (0.3, -0.2, 0.0), bbox=(400, 100, 500, 300))
VERSO_A = (0.3, 0.2, -0.3)    # direzione utente = verso A
VERSO_B = (0.3, -0.2, -0.3)


def run(arb, cycles, candidates, direction, t0=0.0, dt=1.0):
    res = None
    for i in range(cycles):
        res = arb.step(candidates, direction, EE, t0 + i * dt)
    return res


def test_unico_candidato_scelto_senza_direzione():
    arb = TargetArbiter()
    res = run(arb, N - 1, [A], None)
    assert res.target_uid is None, "prima di N cicli non deve scegliere"
    res = arb.step([A], None, EE, 10.0)
    assert res.target_uid == 'A' and res.reason == "unico candidato"


def test_due_oggetti_sceglie_quello_verso_cui_vai():
    arb = TargetArbiter()
    res = run(arb, N, [A, B], VERSO_A)
    assert res.target_uid == 'A', res
    assert res.scores['A'] > res.scores['B']


def test_utente_fermo_tiene_la_scelta():
    arb = TargetArbiter()
    run(arb, N, [A, B], VERSO_A)
    res = run(arb, 5, [A, B], (0.0, 0.0, 0.0), t0=100.0)
    assert res.target_uid == 'A', "fermandosi deve tenere l'ultima scelta"


def test_utente_fermo_non_sceglie():
    arb = TargetArbiter()
    res = run(arb, 2 * N, [A, B], (0.0, 0.0, 0.0))
    assert res.reason == "utente fermo" and res.target_uid is None


def test_caso_ambiguo_non_sceglie():
    # Due oggetti quasi nella stessa direzione: punteggi troppo vicini.
    A2 = cand('A2', (0.30, 0.01, 0.0))
    B2 = cand('B2', (0.30, -0.01, 0.0), bbox=(300, 0, 400, 100))
    arb = TargetArbiter()
    res = run(arb, 2 * N, [A2, B2], (0.3, 0.0, -0.3))
    assert res.reason == "ambiguo" and res.target_uid is None, res


def test_oggetto_alle_spalle_non_scelto():
    arb = TargetArbiter()
    res = run(arb, 2 * N, [A, B], (-0.3, 0.0, 0.0))   # ci si allontana da entrambi
    assert res.target_uid is None and res.reason == "ambiguo"


def test_dopo_la_scelta_non_cambia_idea():
    arb = TargetArbiter(ArbiterParams(direction_tau=0.01))   # filtro quasi istantaneo
    run(arb, N, [A, B], VERSO_A)
    # Anche molti cicli verso B non cambiano la scelta: e' bloccata come un `t`.
    res = run(arb, 5 * N, [A, B], VERSO_B, t0=50.0)
    assert res.target_uid == 'A' and res.reason == "scelta bloccata", res
    # Occlusione da vicino: A non piu' visto, resta solo B in vista.
    res = run(arb, 5 * N, [B], VERSO_B, t0=100.0)
    assert res.target_uid == 'A', "l'occlusione del target non deve spostare la scelta"


def test_prima_della_scelta_serve_conferma():
    arb = TargetArbiter(ArbiterParams(direction_tau=0.01))
    # Proposte alternate A/B: la conferma riparte ogni volta, nessuna scelta.
    for i in range(3 * N):
        arb.step([A, B], VERSO_A if i % 2 == 0 else VERSO_B, EE, float(i))
    assert arb.target_uid is None


def test_reset_permette_una_nuova_scelta():
    arb = TargetArbiter(ArbiterParams(direction_tau=0.01))
    run(arb, N, [A, B], VERSO_A)
    arb.reset()
    res = run(arb, N, [A, B], VERSO_B, t0=50.0)
    assert res.target_uid == 'B', res


def test_parte_contenuta_scartata():
    bottle = cand('bottle', (0.3, 0.0, 0.0), bbox=(100, 100, 200, 400))
    cap = cand('cap', (0.3, 0.0, 0.05), bbox=(130, 100, 170, 140), label='cap')
    assert containment(cap['bbox'], bottle['bbox']) == 1.0
    assert [c['uid'] for c in drop_contained([bottle, cap], 0.8)] == ['bottle']
    arb = TargetArbiter()
    res = run(arb, N, [bottle, cap], None)
    assert res.target_uid == 'bottle' and res.reason == "unico candidato"


def test_soppressione_spegnibile():
    bottle = cand('bottle', (0.3, 0.0, 0.0), bbox=(100, 100, 200, 400))
    cap = cand('cap', (0.3, 0.0, 0.05), bbox=(130, 100, 170, 140), label='cap')
    arb = TargetArbiter(ArbiterParams(suppress_contained=False))
    res = run(arb, N, [bottle, cap], None)
    assert res.reason == "utente fermo", "senza soppressione restano due candidati"


def test_doppione_stessa_etichetta_distinto_per_uid():
    A_bis = cand('A', (0.3, 0.2, 0.0), bbox=(100, 100, 200, 300), label='bottle')
    B_bis = cand('B', (0.3, -0.2, 0.0), bbox=(400, 100, 500, 300), label='bottle')
    arb = TargetArbiter()
    res = run(arb, N, [A_bis, B_bis], VERSO_B)
    assert res.target_uid == 'B', "due 'bottle': la scelta deve seguire l'uid"


def test_nessun_candidato_tiene_la_scelta():
    arb = TargetArbiter()
    run(arb, N, [A, B], VERSO_A)
    res = run(arb, 3, [], VERSO_A, t0=200.0)
    assert res.target_uid == 'A'
    assert TargetArbiter().step([], VERSO_A, EE, 0.0).reason == "nessun candidato"


def test_reset():
    arb = TargetArbiter()
    run(arb, N, [A, B], VERSO_A)
    arb.reset()
    assert arb.target_uid is None


def test_filtro_direzione_tempo_reale():
    # Stessa costante di tempo, cicli a 1s e a 5s: dopo 10s reali di comando
    # costante il filtro deve essere vicino al comando in entrambi i casi.
    for dt in (1.0, 5.0):
        f = DirectionFilter(tau=2.0)
        f.update((0.0, 0.0, 0.0), 0.0)
        t, v = 0.0, None
        while t < 10.0:
            t += dt
            v = f.update((1.0, 0.0, 0.0), t)
        assert v[0] > 0.9, (dt, v)


def test_filtro_direzione_dt_non_positivo():
    f = DirectionFilter(tau=1.0)
    f.update((1.0, 0.0, 0.0), 5.0)
    v = f.update((100.0, 0.0, 0.0), 5.0)      # stesso istante
    assert v[0] == 1.0
    v = f.update((100.0, 0.0, 0.0), 4.0)      # tempo all'indietro
    assert v[0] == 1.0


def test_punteggi_allineamento():
    s = alignment_scores(VERSO_A, EE, [A, B])
    assert abs(s['A'] - 1.0) < 1e-9 and s['B'] < s['A']


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
