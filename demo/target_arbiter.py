"""
Arbitro del target per lo Stadio C: fra piu' oggetti visti, sceglie quello
verso cui l'utente si sta dirigendo col Falcon.

Vive lato percezione e non dipende da ROS: riceve candidati gia' portati in
base_link e la posizione del polso nel tempo, restituisce l'uid del target
scelto. Il chiamante (sgg_ros_node.py) pubblica il vincitore sul normale
canale a target singolo (/sgg/target_point), come dopo un comando `t`:
ET_node non cambia e il canale multi-candidato resta spento (vedi
Metodologia, 23-24/09).

Ogni pezzo e' una funzione pura e si puo' spegnere dai parametri. La scelta
avviene una volta sola, poi resta bloccata come dopo un `t`.

Revisione del 26/09/2026, dopo le prove in ombra del 25/09:
- la direzione dell'utente e' lo spostamento del polso negli ultimi
  direction_window secondi, non "comando Falcon - posizione robot": il robot
  segue il Falcon da vicino e quella differenza era quasi sempre nulla
  (mediana 0.000 m sui 260 cicli del 25/09);
- un oggetto solo in vista non viene piu' scelto automaticamente: serve che
  l'utente si muova verso di lui, come con piu' oggetti (il 25/09 l'arbitro
  si bloccava sul primo oggetto visto e non cambiava piu');
- la scelta si sblocca se l'oggetto scelto non e' piu' nel grafo;
- l'allineamento si misura nel piano orizzontale (x, y di base_link): con
  il polso 40 cm sopra oggetti distanti 20 cm, in 3D entrambi sono
  soprattutto "in basso" e i punteggi restavano troppo vicini (1.00 contro
  0.91 anche andando dritti verso uno dei due).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArbiterParams:
    # Coseno minimo fra direzione dell'utente e direzione robot->oggetto
    # (0.7 ~ 45 gradi). Valori di partenza non tarati, da provare in lab.
    min_alignment: float = 0.7
    # Il migliore deve superare il secondo almeno di questo, altrimenti ambiguo.
    min_margin: float = 0.1
    # Finestra [s] su cui si misura lo spostamento del polso.
    direction_window: float = 2.0
    # Sotto questo spostamento [m] nella finestra l'utente e' considerato fermo.
    min_direction_norm: float = 0.02
    # Direzione, vettori robot->oggetto e soglia di fermo solo su x e y.
    horizontal_only: bool = True
    # Cicli consecutivi con la stessa proposta prima di scegliere.
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


def alignment_scores(direction, ee_position, candidates, horizontal_only=False):
    """uid -> coseno fra `direction` e (posizione candidato - ee_position),
    eventualmente solo sulle componenti x, y."""
    mask = np.array([1.0, 1.0, 0.0]) if horizontal_only else np.ones(3)
    d = np.asarray(direction, dtype=float) * mask
    dn = np.linalg.norm(d)
    scores = {}
    for c in candidates:
        v = (np.asarray(c['position_base'], dtype=float) - np.asarray(ee_position, dtype=float)) * mask
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


class MotionDirection:
    """Spostamento del polso negli ultimi `window` secondi (tempo reale)."""

    def __init__(self, window):
        self.window = window
        self.samples = deque()   # (t, posizione)

    def observe(self, position, t):
        if self.samples and t <= self.samples[-1][0]:
            return               # orologio fermo o all'indietro: campione ignorato
        self.samples.append((t, np.asarray(position, dtype=float)))
        # Tiene un solo campione piu' vecchio della finestra, come riferimento.
        while len(self.samples) > 2 and self.samples[1][0] <= t - self.window:
            self.samples.popleft()

    def direction(self, now):
        """Vettore fra il campione piu' vecchio nella finestra e il piu' recente,
        o None se non ci sono abbastanza campioni recenti."""
        if len(self.samples) < 2 or now - self.samples[-1][0] > self.window:
            return None
        return self.samples[-1][1] - self.samples[0][1]

    def clear(self):
        self.samples.clear()


class TargetArbiter:
    def __init__(self, params=None):
        self.p = params or ArbiterParams()
        self.motion = MotionDirection(self.p.direction_window)
        self.target_uid = None
        self._last_proposal = None
        self._streak = 0

    def reset(self):
        """Da chiamare quando arriva un comando esplicito o dopo un grasp."""
        self.target_uid = None
        self._last_proposal = None
        self._streak = 0

    def observe_ee(self, position, t):
        """Posizione del polso in base_link; da chiamare spesso (es. ogni 50 ms)."""
        self.motion.observe(position, t)

    def step(self, candidates, ee_position, now, known_uids=None):
        """
        candidates: lista di dict con 'uid', 'label', 'bbox' (x1,y1,x2,y2 pixel)
            e 'position_base' (3 valori, base_link) -- solo oggetti visti ora.
        ee_position: posizione attuale del polso (base_link).
        known_uids: uid di tutti i nodi ancora nel grafo (visti o in memoria);
            se la scelta non e' fra questi, si sblocca.
        """
        direction = self.motion.direction(now)
        if direction is not None and self.p.horizontal_only:
            direction = direction * np.array([1.0, 1.0, 0.0])
        dnorm = float(np.linalg.norm(direction)) if direction is not None else 0.0

        if self.target_uid is not None and known_uids is not None and self.target_uid not in known_uids:
            self.reset()
            forgotten = True
        else:
            forgotten = False

        # Scelta una volta sola (24/09/2026): dopo la selezione l'arbitro si
        # comporta come un comando `t` e non cambia piu' idea -- niente cambi
        # di target nei secondi prima della zona di grasp, quando ET_node sta
        # riempiendo la media delle letture da congelare, e niente cambi per
        # occlusione da vicino. Si riparte con reset() (`x` o grasp) o quando
        # l'oggetto scelto viene dimenticato.
        if self.target_uid is not None:
            return ArbiterResult(self.target_uid, None, "scelta bloccata", {}, dnorm)

        if self.p.suppress_contained:
            candidates = drop_contained(candidates, self.p.contained_ratio)

        scores = {}
        if not candidates:
            proposal, reason = None, "nessun candidato"
        elif dnorm < self.p.min_direction_norm:
            proposal, reason = None, "utente fermo"
        else:
            # Stesso criterio con uno o piu' candidati: l'utente deve andare
            # verso l'oggetto (con uno solo il margine non conta).
            scores = alignment_scores(direction, ee_position, candidates, self.p.horizontal_only)
            proposal = pick_winner(scores, self.p.min_alignment, self.p.min_margin)
            reason = "allineato" if proposal is not None else "ambiguo"

        if forgotten:
            reason = "scelta dimenticata, " + reason

        if proposal is None:
            # Nessuna proposta: la conferma riparte da capo.
            self._last_proposal, self._streak = None, 0
        else:
            self._streak = self._streak + 1 if proposal == self._last_proposal else 1
            self._last_proposal = proposal
            if self._streak >= self.p.hysteresis_cycles:
                self.target_uid = proposal

        return ArbiterResult(self.target_uid, proposal, reason, scores, dnorm)
