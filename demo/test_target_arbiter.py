#!/usr/bin/env python3
"""
Test offline dell'arbitro del target (Stadio C), senza ROS ne' robot.
Importa il modulo vero (demo/target_arbiter.py), non una copia della logica.
Il movimento del polso e' simulato campione per campione nel tempo.

Uso: python3 demo/test_target_arbiter.py   (oppure pytest demo/test_target_arbiter.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from target_arbiter import (ArbiterParams, MotionDirection, TargetArbiter,  # noqa: E402
                            alignment_scores, containment, drop_contained)

EE0 = np.array([0.0, 0.0, 0.3])   # posizione iniziale del polso in base_link [m]
N = ArbiterParams().hysteresis_cycles


def cand(uid, pos, bbox=(0, 0, 100, 100), label='bottle'):
    return {'uid': uid, 'label': label, 'bbox': bbox, 'position_base': pos}


# Due oggetti: A davanti-sinistra, B davanti-destra.
A = cand('A', (0.3, 0.2, 0.0), bbox=(100, 100, 200, 300))
B = cand('B', (0.3, -0.2, 0.0), bbox=(400, 100, 500, 300))


def unit_towards(target, start=EE0):
    v = np.asarray(target, dtype=float) - start
    return v / np.linalg.norm(v)


def run(arb, candidates, velocity, cycles, start=EE0, t0=0.0, cycle_dt=1.0, known_uids=None):
    """Muove il polso a velocita' costante [m/s] e chiama step() una volta per
    ciclo (ogni cycle_dt secondi), con campioni del polso ogni 50 ms.
    Restituisce l'ultimo risultato e la posizione finale."""
    velocity = np.asarray(velocity, dtype=float)
    p, t, res = np.array(start, dtype=float), t0, None
    arb.observe_ee(p, t)
    for _ in range(cycles):
        for _ in range(int(round(cycle_dt / 0.05))):
            t += 0.05
            p = p + velocity * 0.05
            arb.observe_ee(p, t)
        res = arb.step(candidates, p, t, known_uids)
    return res, p, t


def test_utente_fermo_non_sceglie_neanche_con_un_solo_oggetto():
    arb = TargetArbiter()
    res, _, _ = run(arb, [A], (0, 0, 0), 3 * N)
    assert res.target_uid is None and res.reason == "utente fermo", res


def test_un_solo_oggetto_scelto_se_ci_si_va_incontro():
    arb = TargetArbiter()
    res, _, _ = run(arb, [A], 0.05 * unit_towards(A['position_base']), N)
    assert res.target_uid == 'A', res


def test_un_solo_oggetto_non_scelto_se_ci_si_allontana():
    arb = TargetArbiter()
    res, _, _ = run(arb, [A], -0.05 * unit_towards(A['position_base']), 3 * N)
    assert res.target_uid is None and res.reason == "ambiguo", res


def test_due_oggetti_sceglie_quello_verso_cui_vai():
    arb = TargetArbiter()
    res, _, _ = run(arb, [A, B], 0.05 * unit_towards(A['position_base']), N)
    assert res.target_uid == 'A', res
    assert res.scores['A'] > res.scores['B']


def test_robot_che_insegue_il_falcon_da_vicino():
    # Il caso del 25/09: il polso si muove (qui 3 cm/s) anche se comando e
    # posizione del robot coincidono; la direzione deve vederlo.
    arb = TargetArbiter()
    res, _, _ = run(arb, [A, B], 0.03 * unit_towards(B['position_base']), N)
    assert res.direction_norm >= 0.04 and res.target_uid == 'B', res


def test_oggetti_vicini_visti_dall_alto():
    # Il caso visto il 26/09 nel nodo: oggetti a 22 cm, polso 40 cm sopra,
    # movimento soprattutto verso il basso ma un po' verso la bottiglia.
    start = np.array([0.0, 0.5, 0.40])
    bottle = cand('bottle', (-0.11, 0.5, 0.0), bbox=(100, 100, 200, 300))
    cup = cand('cup', (0.11, 0.5, 0.0), bbox=(400, 100, 500, 300), label='cup')
    arb = TargetArbiter()
    # Un ciclo in piu': lo spostamento laterale e' lento (circa 1 cm/s) e nel
    # primo ciclo la finestra di 2 s non e' ancora piena.
    res, _, _ = run(arb, [bottle, cup], 0.04 * unit_towards(bottle['position_base'], start), N + 1, start=start)
    assert res.target_uid == 'bottle', res
    # Dalla posizione iniziale, in 3D i due oggetti si separano appena (margine
    # 0.14, vicino alla soglia 0.1; nel nodo con la geometria reale era 0.09),
    # in orizzontale sono opposti: e' il motivo della scelta.
    d = unit_towards(bottle['position_base'], start)
    s3d = alignment_scores(d, start, [bottle, cup])
    s2d = alignment_scores(d, start, [bottle, cup], horizontal_only=True)
    assert s3d['bottle'] - s3d['cup'] < 0.2, s3d
    assert s2d['bottle'] - s2d['cup'] > 1.9, s2d


def test_discesa_verticale_non_decide():
    # Scendere dritti non dice verso quale oggetto si va.
    arb = TargetArbiter()
    res, _, _ = run(arb, [A, B], (0, 0, -0.05), 3 * N)
    assert res.target_uid is None and res.reason == "utente fermo", res


def test_caso_ambiguo_non_sceglie():
    # Due oggetti quasi nella stessa direzione: punteggi troppo vicini.
    A2 = cand('A2', (0.30, 0.01, 0.0))
    B2 = cand('B2', (0.30, -0.01, 0.0), bbox=(300, 0, 400, 100))
    arb = TargetArbiter()
    res, _, _ = run(arb, [A2, B2], 0.05 * unit_towards((0.3, 0.0, 0.0)), 3 * N)
    assert res.reason == "ambiguo" and res.target_uid is None, res


def test_dopo_la_scelta_non_cambia_idea():
    arb = TargetArbiter()
    _, p, t = run(arb, [A, B], 0.05 * unit_towards(A['position_base']), N)
    res, p, t = run(arb, [A, B], 0.05 * unit_towards(B['position_base'], p), 3 * N, start=p, t0=t)
    assert res.target_uid == 'A' and res.reason == "scelta bloccata", res
    # Occlusione da vicino: A non piu' visto ma ancora nel grafo (in memoria).
    res, _, _ = run(arb, [B], (0, 0, 0), 3, start=p, t0=t, known_uids={'A', 'B'})
    assert res.target_uid == 'A', "l'occlusione del target non deve spostare la scelta"


def test_scelta_dimenticata_si_sblocca():
    arb = TargetArbiter()
    _, p, t = run(arb, [A, B], 0.05 * unit_towards(A['position_base']), N)
    assert arb.target_uid == 'A'
    res, _, _ = run(arb, [B], 0.05 * unit_towards(B['position_base'], p), 1, start=p, t0=t, known_uids={'B'})
    assert res.reason.startswith("scelta dimenticata") and res.target_uid is None, res
    res, _, _ = run(arb, [B], 0.05 * unit_towards(B['position_base'], p), N, start=p, t0=t + 1.0, known_uids={'B'})
    assert res.target_uid == 'B', res


def test_prima_della_scelta_serve_conferma():
    arb = TargetArbiter()
    p, t = EE0, 0.0
    # Direzione che cambia a ogni ciclo: la conferma riparte ogni volta.
    for i in range(3 * N):
        target = A['position_base'] if i % 2 == 0 else B['position_base']
        _, p, t = run(arb, [A, B], 0.08 * unit_towards(target, p), 1, start=p, t0=t, cycle_dt=2.0)
    assert arb.target_uid is None


def test_reset_permette_una_nuova_scelta():
    arb = TargetArbiter()
    _, p, t = run(arb, [A, B], 0.05 * unit_towards(A['position_base']), N)
    arb.reset()
    res, _, _ = run(arb, [A, B], 0.08 * unit_towards(B['position_base'], p), N + 1, start=p, t0=t)
    assert res.target_uid == 'B', res


def test_parte_contenuta_scartata():
    bottle = cand('bottle', (0.3, 0.0, 0.0), bbox=(100, 100, 200, 400))
    cap = cand('cap', (0.3, 0.0, 0.05), bbox=(130, 100, 170, 140), label='cap')
    assert containment(cap['bbox'], bottle['bbox']) == 1.0
    assert [c['uid'] for c in drop_contained([bottle, cap], 0.8)] == ['bottle']
    arb = TargetArbiter()
    res, _, _ = run(arb, [bottle, cap], 0.05 * unit_towards(bottle['position_base']), N)
    assert res.target_uid == 'bottle', res


def test_soppressione_spegnibile():
    bottle = cand('bottle', (0.3, 0.0, 0.0), bbox=(100, 100, 200, 400))
    cap = cand('cap', (0.3, 0.0, 0.05), bbox=(130, 100, 170, 140), label='cap')
    arb = TargetArbiter(ArbiterParams(suppress_contained=False))
    res, _, _ = run(arb, [bottle, cap], 0.05 * unit_towards(bottle['position_base']), 3 * N)
    assert res.target_uid is None and res.reason == "ambiguo", "senza soppressione i due sono indistinguibili"


def test_doppione_stessa_etichetta_distinto_per_uid():
    A_bis = cand('A', (0.3, 0.2, 0.0), bbox=(100, 100, 200, 300), label='bottle')
    B_bis = cand('B', (0.3, -0.2, 0.0), bbox=(400, 100, 500, 300), label='bottle')
    arb = TargetArbiter()
    res, _, _ = run(arb, [A_bis, B_bis], 0.05 * unit_towards(B_bis['position_base']), N)
    assert res.target_uid == 'B', "due 'bottle': la scelta deve seguire l'uid"


def test_nessun_candidato():
    arb = TargetArbiter()
    _, p, t = run(arb, [A, B], 0.05 * unit_towards(A['position_base']), N)
    res, _, _ = run(arb, [], (0, 0, 0), 3, start=p, t0=t, known_uids={'A', 'B'})
    assert res.target_uid == 'A'
    res, _, _ = run(TargetArbiter(), [], 0.05 * unit_towards(A['position_base']), 1)
    assert res.reason == "nessun candidato"


def test_direzione_su_finestra_temporale():
    m = MotionDirection(window=2.0)
    for i in range(101):                       # 5 s a 1 cm/s, campioni ogni 50 ms
        m.observe((0.01 * i * 0.05, 0.0, 0.0), i * 0.05)
    d = m.direction(5.0)
    assert abs(d[0] - 0.02) < 0.002, d         # solo gli ultimi ~2 s: circa 2 cm


def test_direzione_assente_se_la_posa_non_arriva():
    m = MotionDirection(window=2.0)
    m.observe((0, 0, 0), 0.0)
    m.observe((0.05, 0, 0), 1.0)
    assert m.direction(10.0) is None           # ultimo campione vecchio di 9 s


def test_direzione_ignora_tempo_all_indietro():
    m = MotionDirection(window=2.0)
    m.observe((0, 0, 0), 5.0)
    m.observe((1.0, 0, 0), 4.0)                # scartato
    m.observe((0.02, 0, 0), 5.5)
    assert abs(m.direction(5.5)[0] - 0.02) < 1e-9


def test_punteggi_allineamento():
    s = alignment_scores(unit_towards(A['position_base']), EE0, [A, B])
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
