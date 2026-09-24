"""
Arbitro del target per lo Stadio C: fra piu' oggetti visti, sceglie quello
verso cui l'utente si sta dirigendo col Falcon.

Vive lato percezione e non dipende da ROS: riceve candidati gia' portati in
base_link e la direzione dell'utente, restituisce l'uid del target scelto.
Il chiamante (sgg_ros_node.py) pubblica il vincitore sul normale canale a
target singolo (/sgg/target_point), come dopo un comando `t`: ET_node non
cambia e il canale multi-candidato resta spento (vedi Metodologia, 23-24/09).

Ogni pezzo e' una funzione pura e si puo' spegnere dai parametri. Con un solo
candidato l'arbitro lo sceglie senza bisogno di direzione (= Stadio B).
La scelta avviene una volta sola, poi resta bloccata come dopo un `t`.
"""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArbiterParams:
    # Coseno minimo fra direzione dell'utente e direzione robot->oggetto
    # (0.7 ~ 45 gradi). Valori di partenza non tarati, da provare in lab.
    min_alignment: float = 0.7
    # Il migliore deve superare il secondo almeno di questo, altrimenti ambiguo.
    min_margin: float = 0.1
    # Sotto questa norma [m] della direzione (comando Falcon - posizione
    # robot) l'utente e' considerato fermo: nessuna nuova decisione.
    min_direction_norm: float = 0.01
    # Costante di tempo [s] del filtro sulla direzione (tempo reale, non cicli).
    direction_tau: float = 0.5
    # Cicli consecutivi con la stessa proposta prima di cambiare target.
    hysteresis_cycles: int = 3
    # Scarta un candidato contenuto quasi tutto nella bbox di uno piu' grande
    # (es. 'cap' del nastro dentro 'bottle').
    suppress_contained: bool = True
    contained_ratio: float = 0.8


@dataclass
class ArbiterResult:
    target_uid: object          # scelta corrente (tenuta anche se l'utente si ferma)
    proposal_uid: object        # proposta di questo ciclo, None se nessuna
    reason: str
    scores: dict = field(default_factory=dict)   # uid -> allineamento
    direction_norm: float = 0.0


def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(inner, outer):
    """Frazione dell'area di `inner` che cade dentro `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    area = bbox_area(inner)
    if area <= 0.0:
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area


def drop_contained(candidates, ratio):
    """Toglie i candidati contenuti (>= ratio) nella bbox di un altro piu' grande."""
    kept = []
    for c in candidates:
        inside_bigger = any(
            o is not c and bbox_area(o['bbox']) > bbox_area(c['bbox'])
            and containment(c['bbox'], o['bbox']) >= ratio
            for o in candidates
        )
        if not inside_bigger:
            kept.append(c)
    return kept


def alignment_scores(direction, ee_position, candidates):
    """uid -> coseno fra `direction` e (posizione candidato - ee_position)."""
    d = np.asarray(direction, dtype=float)
    dn = np.linalg.norm(d)
    scores = {}
    for c in candidates:
        v = np.asarray(c['position_base'], dtype=float) - np.asarray(ee_position, dtype=float)
        vn = np.linalg.norm(v)
        scores[c['uid']] = float(d.dot(v) / (dn * vn)) if dn > 0.0 and vn > 0.0 else -1.0
    return scores


def pick_winner(scores, min_alignment, min_margin):
    """uid del migliore se sopra soglia e staccato dal secondo, altrimenti None."""
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_uid, best = ranked[0]
    if best < min_alignment:
        return None
    if len(ranked) > 1 and best - ranked[1][1] < min_margin:
        return None
    return best_uid


class DirectionFilter:
    """Media esponenziale su tempo reale: alpha = exp(-dt/tau)."""

    def __init__(self, tau):
        self.tau = tau
        self.value = None
        self.last_t = None

    def update(self, raw, t):
        raw = np.asarray(raw, dtype=float)
        if self.value is None:
            self.value, self.last_t = raw, t
            return self.value
        dt = t - self.last_t
        if dt <= 0.0:
            # Orologio fermo o all'indietro: tieni il valore precedente.
            return self.value
        alpha = math.exp(-dt / self.tau)
        self.value = alpha * self.value + (1.0 - alpha) * raw
        self.last_t = t
        return self.value


class TargetArbiter:
    def __init__(self, params=None):
        self.p = params or ArbiterParams()
        self.direction = DirectionFilter(self.p.direction_tau)
        self.target_uid = None
        self._last_proposal = None
        self._streak = 0

    def reset(self):
        """Da chiamare quando arriva un comando esplicito o dopo un grasp."""
        self.target_uid = None
        self._last_proposal = None
        self._streak = 0

    def step(self, candidates, raw_direction, ee_position, now):
        """
        candidates: lista di dict con 'uid', 'label', 'bbox' (x1,y1,x2,y2 pixel)
            e 'position_base' (3 valori, base_link) -- solo oggetti visti ora.
        raw_direction: comando Falcon - posizione robot (base_link), o None.
        """
        direction = None
        if raw_direction is not None:
            direction = self.direction.update(raw_direction, now)
        dnorm = float(np.linalg.norm(direction)) if direction is not None else 0.0

        # Scelta una volta sola (24/09/2026): dopo la selezione l'arbitro si
        # comporta come un comando `t` e non cambia piu' idea -- niente cambi
        # di target nei secondi prima della zona di grasp, quando ET_node sta
        # riempiendo la media delle letture da congelare, e niente cambi per
        # occlusione da vicino. Si riparte solo con reset() (`x` o grasp).
        if self.target_uid is not None:
            return ArbiterResult(self.target_uid, None, "scelta bloccata", {}, dnorm)

        if self.p.suppress_contained:
            candidates = drop_contained(candidates, self.p.contained_ratio)

        scores = {}
        if not candidates:
            proposal, reason = None, "nessun candidato"
        elif len(candidates) == 1:
            proposal, reason = candidates[0]['uid'], "unico candidato"
        elif dnorm < self.p.min_direction_norm:
            proposal, reason = None, "utente fermo"
        else:
            scores = alignment_scores(direction, ee_position, candidates)
            proposal = pick_winner(scores, self.p.min_alignment, self.p.min_margin)
            reason = "allineato" if proposal is not None else "ambiguo"

        if proposal is None:
            # Nessuna proposta: la conferma riparte da capo.
            self._last_proposal, self._streak = None, 0
        else:
            self._streak = self._streak + 1 if proposal == self._last_proposal else 1
            self._last_proposal = proposal
            if self._streak >= self.p.hysteresis_cycles:
                self.target_uid = proposal

        return ArbiterResult(self.target_uid, proposal, reason, scores, dnorm)
