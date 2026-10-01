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
    # Coseno minimo fra direzione dell'utente e direzione robot->oggetto.
    # Valori tarati in laboratorio il 29/09/2026 (prove 7 quater-septies):
    # soglia 0.6 (~53 gradi), conferma su 2 cicli, finestra di 4 s.
    min_alignment: float = 0.6
    # Il migliore deve superare il secondo almeno di questo, altrimenti ambiguo.
    min_margin: float = 0.1
    # Finestra [s] su cui si misura lo spostamento del polso.
    direction_window: float = 4.0
    # Sotto questo spostamento [m] nella finestra l'utente e' considerato fermo.
    min_direction_norm: float = 0.02
    # Direzione, vettori robot->oggetto e soglia di fermo solo su x e y.
    horizontal_only: bool = True
    # Cicli consecutivi con la stessa proposta prima di scegliere.
    hysteresis_cycles: int = 2
    # Scarta un candidato contenuto quasi tutto nella bbox di uno piu' grande
    # (es. 'cap' del nastro dentro 'bottle').
    suppress_contained: bool = True
    contained_ratio: float = 0.8
    # A parita' di direzione vince il piu' vicino (30/09/2026): con gli
    # oggetti in fila lungo la direzione di movimento (vista inclinata)
    # l'allineamento non li distingue (bottle 1.00, cup 0.98, plate 0.93).
    # Fra i candidati sopra soglia ed entro min_margin dal migliore si
    # sceglie il piu' vicino al polso nel piano, se lo e' di almeno
    # nearest_margin [m] rispetto al successivo.
    prefer_nearest: bool = True
    nearest_margin: float = 0.05
    # Discesa (01/10/2026): chi vuole prendere un oggetto si mette sopra il
    # suo punto di avvicinamento e scende. Se il polso e' sceso di almeno
    # descent_min [m] nella finestra e un solo oggetto ha il punto di
    # avvicinamento (posizione - R_pinza * [0, 0, gripper_offset]) entro
    # descent_radius in orizzontale dal polso (il piu' vicino, staccato di
    # nearest_margin dal secondo), la proposta e' quell'oggetto. Con gli
    # oggetti in fila la sola direzione orizzontale non basta (30/09).
    use_descent: bool = True
    descent_min: float = 0.02
    descent_radius: float = 0.10
    gripper_offset: float = 0.184


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


def pick_nearest_among_aligned(scores, candidates, ee_position, min_alignment, min_margin,
                               nearest_margin, horizontal_only=True):
    """Come pick_winner, ma se piu' candidati sono allineati quasi allo
    stesso modo sceglie il piu' vicino al polso (se lo e' di almeno
    nearest_margin rispetto al secondo piu' vicino). None se ambiguo."""
    if not scores:
        return None
    best = max(scores.values())
    if best < min_alignment:
        return None
    aligned = [u for u, sc in scores.items() if sc >= min_alignment and best - sc < min_margin]
    if len(aligned) == 1:
        return aligned[0]
    mask = np.array([1.0, 1.0, 0.0]) if horizontal_only else np.ones(3)
    pos = {c['uid']: np.asarray(c['position_base'], dtype=float) for c in candidates}
    ee = np.asarray(ee_position, dtype=float)
    dist = sorted((float(np.linalg.norm((pos[u] - ee) * mask)), u) for u in aligned)
    if dist[1][0] - dist[0][0] >= nearest_margin:
        return dist[0][1]
    return None


def pick_by_descent(candidates, ee_position, ee_rotation, gripper_offset, radius, nearest_margin):
    """uid dell'unico oggetto il cui punto di avvicinamento e' entro radius
    in orizzontale dal polso (con stacco nearest_margin dal secondo), o None."""
    ee = np.asarray(ee_position, dtype=float)
    R = np.eye(3) if ee_rotation is None else np.asarray(ee_rotation, dtype=float)
    off = R @ np.array([0.0, 0.0, gripper_offset])
    dist = []
    for c in candidates:
        approach = np.asarray(c['position_base'], dtype=float) - off
        dist.append((float(np.linalg.norm((approach - ee)[:2])), c['uid']))
    dist.sort()
    if not dist or dist[0][0] > radius:
        return None
    if len(dist) > 1 and dist[1][0] - dist[0][0] < nearest_margin:
        return None
    return dist[0][1]


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


class CandidateMemory:
    """Posizioni in base_link degli oggetti visti di recente (29/09/2026).

    Con la camera in movimento il rilevatore perde gli oggetti per diversi
    cicli: nelle prove del 29/09 in 13 cicli su 17 con il polso in movimento
    non c'era nessun candidato "visto ora". La posizione in base_link di un
    oggetto fermo non cambia con la camera, quindi resta valida finche' e'
    recente (max_age secondi dall'ultima volta che e' stato visto).

    I candidati contenuti nel riquadro di uno piu' grande (es. 'cap' dentro
    'bottle') si scartano sui soli oggetti visti nello stesso ciclo, dove i
    riquadri sono confrontabili, e in quel ciclo si tolgono anche dalla
    memoria. Non restano esclusi per sempre: un riquadro grande occasionale
    (es. 'person') non deve cancellare un oggetto vero per tutta la prova.

    La memoria e' legata alla posizione, non ai nomi (29/09/2026): se il
    grafo toglie un nodo e lo stesso oggetto ricompare con un altro uid o
    un'altra etichetta (banana vista come 'bird' o 'vase'), il nuovo nodo
    nello stesso punto prende il posto del vecchio (merge_distance). Per
    questo il chiamante puo' non passare known_uids e tenere in memoria
    anche i nodi appena usciti dal grafo.
    """

    def __init__(self, max_age, contained_ratio=None, merge_distance=0.05):
        self.max_age = max_age
        self.contained_ratio = contained_ratio
        # Due nodi del grafo per lo stesso oggetto fisico (29/09: bottle#4 e
        # bottle#5 nello stesso punto) darebbero punteggi uguali e margine
        # nullo: sotto questa distanza orizzontale [m] si tiene il piu' recente.
        self.merge_distance = merge_distance
        self.entries = {}        # uid -> (t, candidato)

    def update(self, seen, now):
        """seen: candidati visti in questo ciclo (stesso formato di step())."""
        kept = seen
        if self.contained_ratio is not None:
            kept = drop_contained(seen, self.contained_ratio)
            kept_uids = {c['uid'] for c in kept}
            for c in seen:
                if c['uid'] not in kept_uids:
                    self.entries.pop(c['uid'], None)
        for c in kept:
            self.entries[c['uid']] = (now, c)

    def candidates(self, now, known_uids=None):
        """Candidati visti negli ultimi max_age secondi e ancora nel grafo."""
        valid = []
        for uid, (t, c) in list(self.entries.items()):
            if now - t > self.max_age or (known_uids is not None and uid not in known_uids):
                del self.entries[uid]
                continue
            valid.append((t, c))
        out = []
        for t, c in sorted(valid, key=lambda tc: tc[0], reverse=True):   # piu' recenti prima
            p = np.asarray(c['position_base'], dtype=float)[:2]
            if any(np.linalg.norm(p - np.asarray(o['position_base'], dtype=float)[:2]) < self.merge_distance
                   for o in out):
                continue
            out.append(c)
        return out

    def clear(self):
        self.entries.clear()


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

    def step(self, candidates, ee_position, now, known_uids=None, ee_rotation=None):
        """
        candidates: lista di dict con 'uid', 'label', 'bbox' (x1,y1,x2,y2 pixel)
            e 'position_base' (3 valori, base_link) -- solo oggetti visti ora.
        ee_position: posizione attuale del polso (base_link).
        known_uids: uid di tutti i nodi ancora nel grafo (visti o in memoria);
            se la scelta non e' fra questi, si sblocca.
        """
        direction = self.motion.direction(now)
        descending = (direction is not None and self.p.use_descent
                      and -float(direction[2]) >= self.p.descent_min)
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
        descent_pick = None
        if candidates and descending:
            descent_pick = pick_by_descent(candidates, ee_position, ee_rotation, self.p.gripper_offset,
                                           self.p.descent_radius, self.p.nearest_margin)
        if not candidates:
            proposal, reason = None, "nessun candidato"
        elif descent_pick is not None:
            proposal, reason = descent_pick, "discesa"
        elif dnorm < self.p.min_direction_norm:
            proposal, reason = None, "utente fermo"
        else:
            # Stesso criterio con uno o piu' candidati: l'utente deve andare
            # verso l'oggetto (con uno solo il margine non conta).
            scores = alignment_scores(direction, ee_position, candidates, self.p.horizontal_only)
            if self.p.prefer_nearest:
                proposal = pick_nearest_among_aligned(scores, candidates, ee_position, self.p.min_alignment,
                                                      self.p.min_margin, self.p.nearest_margin,
                                                      self.p.horizontal_only)
            else:
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
