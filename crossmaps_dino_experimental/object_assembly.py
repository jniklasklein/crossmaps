"""Reversible object hypotheses over stable surface-continuity entity IDs.

No geometry or descriptors are merged. Only jointly observed endpoints vote;
rendering and occlusion never create evidence. Grouping remains a hypothesis:
appearance/contact cannot prove physical identity for arbitrary touching objects.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations, product
import math


def similarity(a, b):
    if not a or not b or len(a) != len(b):
        return None
    norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    value = sum(x*y for x, y in zip(a, b)) / norm if norm > 1e-12 else None
    return value if value is not None and math.isfinite(value) else None


@dataclass
class AssemblyConfig:
    enabled: bool = True
    sensitivity: float = 0.5
    confirm_observations: int = 3
    separate_observations: int = 3
    max_extent_m: float = 3.5
    max_members: int = 12
    max_pairs: int = 256
    group_min_dino: float = .55
    min_contact_fraction: float = .04
    min_contact_samples: int = 2

    @property
    def min_dino(self):
        return 0.92 - 0.24 * min(1.0, max(0.0, self.sensitivity))

    @property
    def max_gap(self):
        return 0.03 + 0.24 * min(1.0, max(0.0, self.sensitivity))


class ObjectAssemblies:
    def __init__(self, config=None):
        self.config = config or AssemblyConfig()
        self.last_seq = -1
        self.pairs = {}
        self.aliases = {}
        self.blocked = set()
        self.budget_limited = False

    @staticmethod
    def key(a, b):
        return tuple(sorted((int(a), int(b))))

    def canonical(self, eid):
        return self.aliases.get(eid, eid) if self.config.enabled else eid

    def separate(self, a, b):
        key = self.key(a, b)
        self.blocked.add(key)
        self.pairs.pop(key, None)
        group = self.canonical(a)
        self.aliases = {eid: owner for eid, owner in self.aliases.items() if owner != group}

    def allow(self, a, b):
        self.blocked.discard(self.key(a, b))
        self.pairs.pop(self.key(a, b), None)  # Must earn fresh evidence again.

    @staticmethod
    def gap(a, b):
        return math.sqrt(sum(max(0.0, a['min'][i]-b['max'][i], b['min'][i]-a['max'][i])**2
                             for i in range(3)))

    def _contact(self, a, b, voxel_size):
        # AABB adjacency alone is not contact (e.g. crossing boxes with no
        # measured surface nearby). Bounded queries into actual voxel support.
        smaller, other = (a, b) if len(a['support']) <= len(b['support']) else (b, a)
        pitch = other['contact_pitch']
        gap_cells_sq = (self.config.max_gap / max(voxel_size, 0.01)) ** 2
        offsets = product(range(-1, 2), repeat=3)
        offsets = list(offsets)
        hits = 0
        for x, y, z in smaller['sample']:
            bx, by, bz = x // pitch, y // pitch, z // pitch
            matched = False
            for dx, dy, dz in offsets:
                for ox, oy, oz in other['contact_index'].get((bx+dx, by+dy, bz+dz), ()):
                    # Distance between voxel *surfaces*, not only AABB overlap
                    # or a Chebyshev neighbourhood that changes with resolution.
                    sx, sy, sz = max(0, abs(x-ox)-1), max(0, abs(y-oy)-1), max(0, abs(z-oz)-1)
                    if sx*sx + sy*sy + sz*sz <= gap_cells_sq + 1e-9:
                        matched = True
                        break
                if matched:
                    break
            hits += int(matched)
        fraction = hits/max(1, len(smaller['sample']))
        a['pair_contact'] = b['pair_contact'] = dict(samples=hits, fraction=fraction)
        return hits >= self.config.min_contact_samples and fraction >= self.config.min_contact_fraction

    def _reason(self, a, b, shared, voxel_size):
        cos = similarity(a['dino'], b['dino'])
        gap = self.gap(a, b)
        if not a.get('geometry_reliable', True) or not b.get('geometry_reliable', True):
            return 'fragmented_geometry', cos, gap, False
        if any(n.get('surface_role') in ('horizontal_support_surface', 'vertical_structural_surface') for n in (a,b)):
            return 'structural_surface_not_object_part', cos, gap, False
        if gap > self.config.max_gap + 1e-6:
            return 'geometry_gap', cos, gap, False
        if max(max(a['max'][i], b['max'][i])-min(a['min'][i], b['min'][i]) for i in range(3)) > self.config.max_extent_m:
            return 'object_extent', cos, gap, False
        if not self._contact(a, b, voxel_size):
            return 'insufficient_surface_contact', cos, gap, False
        if cos is None:
            return 'missing_dino', cos, gap, False
        if not a.get('fresh_dino', True) or not b.get('fresh_dino', True):
            return 'stale_dino', cos, gap, False
        if cos < self.config.min_dino:
            return 'low_dino', cos, gap, False
        ea = sorted(max(voxel_size, hi-lo) for lo, hi in zip(a['min'], a['max']))
        eb = sorted(max(voxel_size, hi-lo) for lo, hi in zip(b['min'], b['max']))
        same_shape = all(min(x, y)/max(x, y) >= 0.7 for x, y in zip(ea, eb))
        # A row of similarly-sized boxes (or identical cushions) is ambiguous.
        # Such pairs need shared-mask evidence from at least two camera cells.
        if same_shape and not shared:
            return 'similar_shape_needs_shared_mask', cos, gap, True
        return 'dino_contact', cos, gap, same_shape

    def update(self, seq, now, nodes, observed, shared_masks=(), voxel_size=0.15, camera=None, exclusions=()):
        if seq <= self.last_seq:
            return False
        self.last_seq = seq
        alive = set(nodes)
        self.blocked.intersection_update({p for p in self.blocked if set(p) <= alive})
        forbidden = self.blocked | {self.key(*p) for p in exclusions}
        self.pairs = {p: state for p, state in self.pairs.items()
                      if set(p) <= alive and p not in forbidden and (state['confirmed'] or now-state['last_seen'] < 60)}
        if not self.config.enabled:
            self.aliases.clear()
            self.pairs.clear()
            return True
        prepared = {}
        pitch = max(1, math.ceil(self.config.max_gap / max(voxel_size, 0.01)) + 1)
        for eid, node in nodes.items():
            support = set(node.get('support', ()))
            ordered = sorted(support)
            index = {}
            for x, y, z in ordered:
                index.setdefault((x//pitch, y//pitch, z//pitch), []).append((x, y, z))
            prepared[eid] = {**node, 'support': support, 'sample': ordered[::max(1, math.ceil(len(ordered)/128))],
                             'contact_index': index, 'contact_pitch': pitch}
        observed = sorted(set(observed) & alive)
        shared_masks = {self.key(*p) for p in shared_masks}
        candidates = [p for p in combinations(observed, 2) if p not in forbidden and
                      (p in self.pairs or self.canonical(p[0]) == self.canonical(p[1]) or
                       self.gap(prepared[p[0]], prepared[p[1]]) <= self.config.max_gap + 1e-6)]
        candidates.sort(key=lambda p: (not self.pairs.get(p, {}).get('confirmed', False),
                                        self.gap(prepared[p[0]], prepared[p[1]]), p))
        self.budget_limited = len(candidates) > self.config.max_pairs
        for pair in candidates[:self.config.max_pairs]:
            reason, cos, gap, same_shape = self._reason(prepared[pair[0]], prepared[pair[1]],
                                                       pair in shared_masks, voxel_size)
            if pair not in self.pairs and len(self.pairs) >= 1024:
                self.budget_limited = True
                continue
            state = self.pairs.setdefault(pair, dict(hits=0, misses=0, confirmed=False, views=[], shared_views=[], first_seq=seq))
            contact = prepared[pair[0]].pop('pair_contact', {})
            prepared[pair[1]].pop('pair_contact', None)
            state.update(last_seen=now, reason=reason, dino_cos=cos, gap_m=gap, group_blocker=None,
                         geometry_basis=prepared[pair[0]].get('geometry_basis','stored_support'),
                         excluded_outlier_voxels=[prepared[e].get('excluded_outlier_voxels',0) for e in pair],
                         contact_samples=contact.get('samples'), contact_fraction=contact.get('fraction'),
                         descriptor_basis='dino_interior', evidence_kind='geometric_appearance_hypothesis')
            if cos is not None and all(prepared[e].get('fresh_dino', True) for e in pair):
                # Also monitor non-touching endpoints of an existing group.
                # They need compatible appearance, not direct pairwise contact.
                state['coherence_misses'] = (state.get('coherence_misses', 0) + 1
                                             if cos < self.config.group_min_dino else 0)
            if reason == 'dino_contact':
                state['hits'] += 1
                state['misses'] = 0
                view = [math.floor(float(v)/0.3) for v in camera] if camera is not None else [0, 0, 0]
                if view not in state['views'] and len(state['views']) < 32:
                    state['views'].append(view)
                shared_views = state.setdefault('shared_views', [])
                if pair in shared_masks and view not in shared_views and len(shared_views) < 32:
                    shared_views.append(view)
                if same_shape and len(shared_views) < 2:
                    state['reason'] = 'similar_shape_needs_second_view'
                elif state['hits'] >= self.config.confirm_observations:
                    if not state['confirmed']:
                        state['confirmed_seq'] = seq
                    state['confirmed'] = True
            elif reason in ('low_dino', 'geometry_gap', 'object_extent', 'insufficient_surface_contact', 'structural_surface_not_object_part'):
                state['hits'] = 0
                state['misses'] += 1
                if state['misses'] >= self.config.separate_observations:
                    state['confirmed'] = False
                    state['views'] = []
                    state['shared_views'] = []
            elif not state['confirmed'] and reason not in ('missing_dino', 'stale_dino'):
                # Missing/ambiguous evidence is not evidence of separation.
                state['hits'] = 0
        self._rebuild(prepared, forbidden)
        return True

    def _rebuild(self, nodes, forbidden):
        previous = {eid: self.canonical(eid) for eid in nodes}
        groups = {eid: {eid} for eid in nodes}
        owner = {eid: eid for eid in nodes}
        edges = sorted((p for p,s in self.pairs.items() if s['confirmed']),
                       key=lambda p: (self.pairs[p].get('confirmed_seq', 0), p))
        for a, b in edges:
            state = self.pairs[(a, b)]
            state['group_blocker'] = None
            ga, gb = owner[a], owner[b]
            if ga == gb:
                continue
            joined = groups[ga] | groups[gb]
            if len(joined) > self.config.max_members or any(set(p) <= joined for p in forbidden):
                state['group_blocker'] = 'member_limit_or_explicit_separation'
                continue
            established = len({previous[e] for e in joined}) == 1
            if not established and max(max(nodes[e]['max'][i] for e in joined)-min(nodes[e]['min'][i] for e in joined)
                   for i in range(3)) > self.config.max_extent_m:
                state['group_blocker'] = 'group_extent'
                continue
            # Complete appearance compatibility, not transitive single-link
            # clustering (A resembles B resembles C does not imply A resembles C).
            compatible = True
            for x in groups[ga]:
                for y in groups[gb]:
                    evidence = self.pairs.get(self.key(x, y), {})
                    if previous[x] == previous[y] and evidence.get('coherence_misses', 0) < self.config.separate_observations:
                        # Retain established compatibility through missing data
                        # or a single noisy observation; never grant a new join.
                        continue
                    cos = similarity(nodes[x]['dino'], nodes[y]['dino'])
                    if cos is None or cos < self.config.group_min_dino:
                        compatible = False
            if not compatible:
                state['group_blocker'] = 'incompatible_group_appearance'
                continue
            canonical = min(joined)
            groups.pop(ga); groups.pop(gb)
            groups[canonical] = joined
            for eid in joined:
                owner[eid] = canonical
        self.aliases = {eid: root for eid, root in owner.items() if eid != root}

    def diagnostics(self):
        shown = sorted(self.pairs.items(), key=lambda item: (not item[1]['confirmed'],
                       -item[1]['hits'], -(item[1].get('dino_cos') or 0.), item[0]))[:64]
        return {'enabled': self.config.enabled, 'sensitivity': self.config.sensitivity,
                'min_dino_cos': round(self.config.min_dino, 3), 'max_gap_m': round(self.config.max_gap, 3),
                'joined_hypotheses': len(self.aliases), 'budget_limited': self.budget_limited,
                'descriptor_basis': 'dino_interior', 'group_min_dino': self.config.group_min_dino,
                'min_contact_fraction': self.config.min_contact_fraction,
                'pair_count': len(self.pairs), 'pairs': [dict(a=a, b=b, **s) for (a,b),s in shown]}

    def to_dict(self):
        return {'version': 2, 'last_seq': self.last_seq, 'aliases': dict(self.aliases),
                'config': asdict(self.config), 'blocked': sorted(self.blocked),
                'pairs': [[a,b,deepcopy(s)] for (a,b),s in self.pairs.items()]}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        if not data:
            return obj
        if data.get('version') not in (1, 2):
            raise ValueError('Unsupported object assembly version')
        obj.config = AssemblyConfig(**data.get('config', {}))
        obj.last_seq = int(data['last_seq'])
        obj.aliases = {int(e): int(root) for e,root in data.get('aliases', {}).items()}
        obj.blocked = {obj.key(*p) for p in data.get('blocked', [])}
        obj.pairs = {obj.key(a,b): deepcopy(state) for a,b,state in data.get('pairs', [])}
        if data['version'] == 1:
            # Old confirmations used context-adjusted features and single-point
            # contact. Retain explicit corrections, but require fresh evidence.
            obj.aliases.clear()
            obj.pairs.clear()
        if any(e <= 0 or root <= 0 or root >= e or root in obj.aliases for e, root in obj.aliases.items()):
            raise ValueError('Invalid object assembly aliases')
        if any(a <= 0 or a >= b for a, b in obj.pairs.keys() | obj.blocked):
            raise ValueError('Invalid object assembly pair')
        return obj
