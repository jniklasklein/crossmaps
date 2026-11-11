"""Observation-driven identity memory, independent of ROS, models and rendering.

An entity is a persistent *object hypothesis*, not a semantic part-of assertion.
Appearance can veto a support-continuity match, but cannot create one. Existing
base entities are never destructively joined. Optional ObjectAssemblies create
reversible display/graph groups using independent DINO and geometry evidence.
All state changes happen on observations or explicit segment lineage events;
serializing/rendering a graph is read-only.
"""

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass, field
import math
from itertools import product
from object_assembly import AssemblyConfig, ObjectAssemblies


def bounded_support(keys, limit):
    keys = set(keys)
    if len(keys) <= limit:
        return keys
    ordered = sorted(keys)
    return {ordered[i * len(ordered) // limit] for i in range(limit)}


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return None
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator > 1e-12 else None


@dataclass
class IdentityConfig:
    birth_observations: int = 3
    membership_observations: int = 2
    min_overlap_voxels: int = 8
    min_overlap_fraction: float = 0.65
    ambiguity_margin: float = 0.20
    min_dino_cos: float = 0.25
    support_limit: int = 24000
    dormant_retention_s: float = 180.0
    view_cell_m: float = 0.30
    neighborhood_tiebreak: bool = True
    neighborhood_overlap_gap: float = .10
    neighborhood_min_dino: float = .80
    voxel_size_m: float = .15
    neighborhood_max_extent_m: float = 3.5


@dataclass
class EntityRecord:
    id: int
    members: set = field(default_factory=set)
    support: set = field(default_factory=set)
    dino: tuple = ()
    dino_prototypes: list = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    observations: int = 0
    last_seq: int = -1
    views: set = field(default_factory=set)


class PersistentEntities:
    def __init__(self, config=None):
        self.config = config or IdentityConfig()
        self.next_id = 1
        self.last_seq = -1
        self.records = {}
        self.segment_entity = {}
        self.pending = {}
        self.lineage = deque(maxlen=256)
        self.split_exclusions = {}  # A disconnected split does not prove part-of.
        self.assembly_exclusions = {}  # Explicit corrections also veto grouping.
        self.support_index = {}
        self.counters = Counter()
        self.assemblies = ObjectAssemblies(AssemblyConfig(enabled=False))

    def _support(self, record, support):
        for key in record.support:
            owners = self.support_index.get(key)
            if owners is not None:
                owners.discard(record.id)
                if not owners:
                    self.support_index.pop(key, None)
        record.support = bounded_support(support, max(1, self.config.support_limit))
        for key in record.support:
            self.support_index.setdefault(key, set()).add(record.id)

    def _new(self, now):
        eid = self.next_id
        self.next_id += 1
        record = EntityRecord(eid, first_seen=now, last_seen=now)
        self.records[eid] = record
        self.counters["births"] += 1
        return record

    def _attach(self, sid, eid, reason):
        previous = self.segment_entity.get(sid)
        if previous is not None and previous != eid:
            self.records[previous].members.discard(sid)
        self.segment_entity[sid] = eid
        self.records[eid].members.add(sid)
        self.pending.pop(sid, None)
        self.lineage.append({"event": "member_of", "segment_id": sid,
                             "entity_id": eid, "reason": reason, "seq": self.last_seq})

    def segment_split(self, parent, child):
        eid = self.segment_entity.get(int(parent))
        if eid is not None:
            self.split_exclusions[int(child)] = eid
        if int(parent) in self.assembly_exclusions:
            self.assembly_exclusions[int(child)] = self.assembly_exclusions[int(parent)]
        self.lineage.append({"event": "split_from", "segment_id": int(child),
                             "parent_segment_id": int(parent), "seq": self.last_seq})
        self.counters["segment_splits"] += 1

    def detach_segment(self, sid, reason="explicit_correction"):
        """Undo a membership without deleting geometry or renumbering others.

        The rejected entity is excluded from automatic reassociation until the
        segment is retired. No underlying segment geometry is modified.
        """
        sid = int(sid)
        eid = self.segment_entity.pop(sid, None)
        if eid is None:
            return False
        self.records[eid].members.discard(sid)
        self.split_exclusions[sid] = eid
        self.assembly_exclusions[sid] = eid
        self.pending.pop(sid, None)
        self.lineage.append({"event": "membership_removed", "segment_id": sid,
                             "entity_id": eid, "reason": reason, "seq": self.last_seq})
        self.counters["membership_removals"] += 1
        return True

    def can_merge_segments(self, a, b):
        # Maintenance must never collapse independent hypotheses. Unassigned
        # fragments also lack sufficient evidence for a destructive merge.
        ea, eb = self.segment_entity.get(a), self.segment_entity.get(b)
        return ea is not None and ea == eb

    def segment_merged(self, keep, drop):
        eid = self.segment_entity.pop(drop, None)
        if eid in self.records:
            self.records[eid].members.discard(drop)
        self.pending.pop(drop, None)
        self.split_exclusions.pop(drop, None)
        self.assembly_exclusions.pop(drop, None)
        self.lineage.append({"event": "merged_into", "segment_id": drop,
                             "target_segment_id": keep, "seq": self.last_seq})
        self.counters["segment_merges"] += 1

    def _candidates(self, sid, node, now):
        support = set(node.get("support", ()))
        counts = Counter()
        for key in support:
            counts.update(self.support_index.get(key, ()))
        candidates = []
        for eid, count in counts.items():
            record = self.records[eid]
            if self.split_exclusions.get(sid) == eid:
                continue
            if now - record.last_seen > self.config.dormant_retention_s:
                continue
            fraction = count / max(1, len(support))
            if count < self.config.min_overlap_voxels:
                continue
            # Keep secondary geometric candidates for ambiguity detection,
            # even when their overlap is below the acceptance threshold.
            similarities = [cosine(node.get("dino", ()), descriptor)
                            for descriptor in (record.dino_prototypes or [record.dino])]
            similarities = [value for value in similarities if value is not None]
            similarity = max(similarities) if similarities else None
            if similarity is not None and similarity < self.config.min_dino_cos:
                continue
            candidates.append((fraction, eid))
        return sorted(candidates, key=lambda pair: (-pair[0], pair[1]))

    def _neighborhood_evidence(self, sid, eid, node, segments, observed):
        """Independent contact with an established, currently observed part.

        Bounded tie-break evidence, not a new appearance-only candidate source.
        Exact overlap is excluded from contact evidence to avoid counting the
        same geometric evidence twice. Similar-sized repeated objects abstain.
        """
        structural = {'horizontal_support_surface', 'vertical_structural_surface'}
        if node.get('surface_role') in structural or not node.get('fresh_dino', False):
            return 0., None
        support = set(node.get('support', ()))
        if not support:
            return 0., None
        shape = tuple(max(k[i] for k in support)-min(k[i] for k in support)+1 for i in range(3))
        record = self.records[eid]
        if record.observations < max(3, self.config.birth_observations):
            return 0., None
        # Only established membership in this candidate, never newly attached
        # members from the current observation or uncertain neighboring groups.
        best = (0., None)
        for neighbor in sorted(record.members & observed)[:32]:
            if neighbor == sid:
                continue
            other = segments[neighbor]
            if other.get('surface_role') in structural or not other.get('fresh_dino', False):
                continue
            cos = cosine(node.get('group_dino', node.get('dino', ())),
                         other.get('group_dino', other.get('dino', ())))
            if cos is None or not math.isfinite(cos) or cos < self.config.neighborhood_min_dino:
                continue
            clip = cosine(node.get('clip', ()), other.get('clip', ()))
            if clip is not None and (not math.isfinite(clip) or clip < .5):
                continue
            other_support = set(other.get('support', ()))
            independent = other_support-support
            query = support-other_support
            if not independent or not query:
                continue
            combined = support | other_support
            if max(max(k[i] for k in combined)-min(k[i] for k in combined)+1 for i in range(3))*self.config.voxel_size_m > self.config.neighborhood_max_extent_m:
                continue
            other_shape = tuple(max(k[i] for k in other_support)-min(k[i] for k in other_support)+1 for i in range(3))
            if all(min(a,b)/max(a,b) >= .7 for a,b in zip(sorted(shape), sorted(other_shape))):
                continue
            # Require actual touching voxel surfaces, not just nearby boxes.
            ordered = sorted(query)
            sample = ordered[::max(1, math.ceil(len(ordered)/128))]
            hits = sum(any((x+dx,y+dy,z+dz) in independent
                           for dx,dy,dz in product((-1,0,1), repeat=3)) for x,y,z in sample)
            fraction = hits/max(1, len(sample))
            if hits >= 2 and fraction >= .04:
                strength = cos*min(1., fraction/.15)
                if strength > best[0]:
                    best = (strength, neighbor)
        return best

    def _resolve_near_tie(self, sid, node, candidates, segments, observed):
        if not self.config.neighborhood_tiebreak or len(candidates) < 2:
            return None, []
        top = candidates[0][0]
        scored = []
        # Hard geometry/appearance exclusions in _candidates still apply.
        # Candidates outside the near-tie window cannot win a context bonus.
        for overlap, eid in candidates[:3]:
            strength, neighbor = (0., None)
            eligible = overlap >= self.config.min_overlap_fraction and top-overlap <= self.config.neighborhood_overlap_gap
            if eligible:
                strength, neighbor = self._neighborhood_evidence(sid, eid, node, segments, observed)
            scored.append(dict(entity_id=eid, overlap=overlap, neighborhood_strength=strength,
                               neighbor_segment_id=neighbor, score=overlap+.15*strength))
        scored.sort(key=lambda row: (-row['score'], row['entity_id']))
        best = scored[0]
        if best['neighborhood_strength'] > 0 and best['score']-scored[1]['score'] >= .03:
            return best['entity_id'], scored
        return None, scored

    def update(self, seq, now, segments, observed, camera_position=None):
        """Consume one accepted observation; segments includes hidden tracks.

        observed must contain only segments actually fused in this observation,
        never all tracks in the map or everything selected for display.
        """
        if int(seq) <= self.last_seq:
            return False
        self.last_seq = int(seq)
        observed = set(observed) & set(segments)
        view = None
        if camera_position is not None:
            cell = max(0.01, self.config.view_cell_m)
            view = tuple(int(math.floor(float(v) / cell)) for v in camera_position)
        for sid in list(self.segment_entity):
            if sid not in segments:
                eid = self.segment_entity.pop(sid)
                self.records[eid].members.discard(sid)
        for sid in set(self.pending) - set(segments):
            self.pending.pop(sid, None)
        for sid in set(self.split_exclusions) - set(segments):
            self.split_exclusions.pop(sid, None)
        for sid in set(self.assembly_exclusions) - set(segments):
            self.assembly_exclusions.pop(sid, None)

        # Match against the previous observation's support. New births within
        # this update cannot pull their same-frame neighbors into an entity.
        decisions = []
        for sid in sorted(observed):
            if sid in self.segment_entity:
                continue
            node = segments[sid]
            if len(node.get("support", ())) < self.config.min_overlap_voxels:
                continue
            pending = self.pending.setdefault(sid, {"observations": 0, "hits": 0, "first_seen": now, "views": [],
                                                   "candidate": None, "reason": "new_surface"})
            pending["observations"] += 1
            views = pending.setdefault("views", [])
            if view is not None and list(view) not in views and len(views) < 128:
                views.append(list(view))
            candidates = self._candidates(sid, node, now)
            best = candidates[0] if candidates else None
            ambiguous = bool(best and len(candidates) > 1 and
                             best[0] - candidates[1][0] < self.config.ambiguity_margin)
            accepted = bool(best and best[0] >= self.config.min_overlap_fraction and not ambiguous)
            target = best[1] if accepted else None
            neighborhood_target, neighborhood_scores = (None, [])
            if ambiguous:
                neighborhood_target, neighborhood_scores = self._resolve_near_tie(sid, node, candidates, segments, observed)
                if neighborhood_target is not None:
                    target, accepted = neighborhood_target, True
            basis = ('neighborhood', neighborhood_scores[0]['neighbor_segment_id']) if neighborhood_target is not None else ('support', None)
            consecutive = pending.get('last_candidate_seq') == seq-1 and tuple(pending.get('candidate_basis', ())) == basis
            pending["hits"] = pending["hits"] + 1 if consecutive and target is not None and target == pending["candidate"] else int(target is not None)
            pending['last_candidate_seq'] = seq
            pending['candidate_basis'] = basis
            pending["candidate"] = target
            pending["candidates"] = [{"entity_id": eid, "overlap": round(score, 4)} for score, eid in candidates[:3]]
            pending['neighborhood_scores'] = neighborhood_scores
            pending["reason"] = "neighborhood_tiebreak" if neighborhood_target is not None else "ambiguous_support" if ambiguous else "support_continuity" if accepted else "insufficient_overlap" if best else "new_surface"
            required = max(3 if neighborhood_target is not None else 1, self.config.membership_observations)
            if accepted and pending["hits"] >= required:
                decisions.append((sid, target))
            elif not candidates and pending["observations"] >= max(1, self.config.birth_observations):
                decisions.append((sid, None))
        for sid, eid in decisions:
            if eid is None:
                record = self._new(now)
                pending = self.pending[sid]
                record.first_seen = float(pending.get("first_seen", now))
                # Birth delay is not evidence loss: these are distinct measured
                # observations of this surface, never sums across fragments.
                record.observations = pending["observations"] - 1
                record.views = {tuple(v) for v in pending.get("views", [])}
                eid = record.id
                reason = "independent_surface_hypothesis"
            else:
                self.counters["support_reassociations"] += 1
                reason = self.pending[sid]['reason']
                if reason == 'neighborhood_tiebreak':
                    self.counters['neighborhood_reassociations'] += 1
            self._attach(sid, eid, reason)

        touched = {self.segment_entity[sid] for sid in observed if sid in self.segment_entity}
        pending_support = set()
        for sid in self.pending:
            pending_support.update(segments[sid].get("support", ()))
        for eid in touched:
            record = self.records[eid]
            record.last_seq = self.last_seq
            record.last_seen = now
            record.observations += 1  # Not the sum of child observations.
            if view is not None and len(record.views) < 128:
                record.views.add(view)
            support = set()
            for sid in record.members:
                support.update(segments[sid].get("support", ()))
            # Do not erase the anchor of a new fragment during its probation
            # just because its former owner's support has already shrunk.
            support.update(record.support & pending_support)
            self._support(record, support)
            # Keep a real descriptor from a representative surface, not an
            # average of unrelated chair legs, upholstery, etc.
            sources = [segments[sid] for sid in sorted(record.members & observed) if segments[sid].get("dino")]
            if sources:
                record.dino = tuple(max(sources, key=lambda n: len(n.get("support", ())))["dino"])
                # Reuse at most four cached surface descriptors; no extra
                # inference. One upholstery descriptor must not veto a leg's
                # valid geometric correspondence just because it looks different.
                for node in sorted(sources, key=lambda n: len(n.get("support", ()))):
                    descriptor = tuple(node["dino"])
                    record.dino_prototypes = [p for p in record.dino_prototypes
                                              if (cosine(p, descriptor) or 0.0) < 0.95]
                    record.dino_prototypes.append(descriptor)
                    record.dino_prototypes = record.dino_prototypes[-4:]

        for eid, record in list(self.records.items()):
            if not record.members and now - record.last_seen > self.config.dormant_retention_s:
                self._support(record, ())
                del self.records[eid]
                self.counters["expired"] += 1
        return True

    def diagnostics(self):
        return {"entities": len(self.records), "assigned_segments": len(self.segment_entity),
                "pending_segments": len(self.pending),
                "ambiguous_segments": sum(p.get("reason") == "ambiguous_support" for p in self.pending.values()),
                "dormant_entities": sum(not r.members for r in self.records.values()),
                "observation_seq": self.last_seq, "counters": dict(self.counters),
                "objectness": "unverified", "area_inference": "deferred",
                "membership_basis": "surface_continuity", "whole_object_grouping": self.assemblies.config.enabled,
                "assembly": self.assemblies.diagnostics()}

    def to_dict(self):
        return {"version": 1, "next_id": self.next_id, "last_seq": self.last_seq,
                "records": [{"id": r.id, "members": sorted(r.members), "support": sorted(r.support),
                             "dino": list(r.dino), "dino_prototypes": [list(p) for p in r.dino_prototypes],
                             "first_seen": r.first_seen, "last_seen": r.last_seen,
                             "observations": r.observations, "last_seq": r.last_seq, "views": sorted(r.views)}
                            for r in self.records.values()],
                "pending": deepcopy(self.pending), "lineage": deepcopy(list(self.lineage)),
                "split_exclusions": dict(self.split_exclusions), "counters": dict(self.counters),
                "assembly_exclusions": dict(self.assembly_exclusions),
                "assemblies": self.assemblies.to_dict()}

    @classmethod
    def from_dict(cls, payload, config=None):
        memory = cls(config)
        if not payload:
            return memory
        if payload.get("version") != 1:
            raise ValueError("Unsupported entity memory version")
        memory.last_seq = int(payload.get("last_seq", -1))
        for item in payload.get("records", []):
            record = EntityRecord(int(item["id"]), set(map(int, item["members"])),
                                  dino=tuple(item["dino"]),
                                  dino_prototypes=[tuple(p) for p in item.get("dino_prototypes", [])][-4:],
                                  first_seen=float(item["first_seen"]),
                                  last_seen=float(item["last_seen"]), observations=int(item["observations"]),
                                  last_seq=int(item["last_seq"]), views={tuple(v) for v in item["views"]})
            memory.records[record.id] = record
            for sid in record.members:
                if sid in memory.segment_entity:
                    raise ValueError("Segment belongs to multiple entities")
                memory.segment_entity[sid] = record.id
            memory._support(record, {tuple(k) for k in item["support"]})
        memory.next_id = max(int(payload.get("next_id", 1)), max(memory.records, default=0) + 1)
        memory.pending = {int(k): deepcopy(v) for k, v in payload.get("pending", {}).items()}
        memory.split_exclusions = {int(k): int(v) for k, v in payload.get("split_exclusions", {}).items()}
        memory.assembly_exclusions = {int(k): int(v) for k, v in payload.get("assembly_exclusions", {}).items()}
        memory.lineage.extend(payload.get("lineage", []))
        memory.counters.update(payload.get("counters", {}))
        if payload.get("assemblies"):
            memory.assemblies = ObjectAssemblies.from_dict(payload["assemblies"])
            references = set(memory.assemblies.aliases) | set(memory.assemblies.aliases.values())
            references.update(e for pair in memory.assemblies.pairs.keys() | memory.assemblies.blocked for e in pair)
            if not references <= set(memory.records):
                raise ValueError("Object assembly references a missing entity")
        return memory


class ObservationEdges:
    """Relation evidence with explicit observation and visibility accounting."""

    def __init__(self):
        self.last_seq = -1
        self.states = {}

    @staticmethod
    def key(a, b, relation):
        if relation not in ("above", "supported_by", "located_in", "member_of") and a > b:
            a, b = b, a
        return (a, b, relation)

    def update(self, seq, edges, observed, alive, max_misses=3, alpha=0.35):
        if int(seq) <= self.last_seq:
            return
        self.last_seq = int(seq)
        observed, alive = set(observed), set(alive)
        seen = set()
        for edge in edges:
            key = self.key(int(edge["from"]), int(edge["to"]), str(edge["relation"]))
            if key in seen or key[0] not in observed or key[1] not in observed:
                continue
            seen.add(key)
            state = self.states.setdefault(key, {"hits": 0, "misses": 0})
            for name in ("strength", "distance", "semantic_cos"):
                value = float(edge.get(name, 0.0))
                state[name] = value if state["hits"] == 0 else (1 - alpha) * state[name] + alpha * value
            state.update(hits=state["hits"] + 1, misses=0, last_seq=self.last_seq,
                         priority=int(edge.get("priority", 0)))
        for key, state in list(self.states.items()):
            if key[0] not in alive or key[1] not in alive:
                del self.states[key]
            elif key not in seen and key[0] in observed and key[1] in observed:
                state["misses"] += 1
                if state["misses"] > max_misses:
                    del self.states[key]

    def snapshot(self, visible, min_observations=2):
        visible = set(visible)
        return [{"from": a, "to": b, "relation": relation, **state}
                for (a, b, relation), state in self.states.items()
                if a in visible and b in visible and state["hits"] >= min_observations]
