"""Observation-driven display hypotheses; never used as fusion evidence.

Each SAM silhouette gets one provisional owner. Votes remain available for
debugging. Entity votes are aggregated *before* selecting the display owner.
"""
import numpy as np


class DominantOverlay:
    def __init__(self, switch_observations=3, switch_margin=.10, max_age_s=15.):
        self.switch_observations = switch_observations
        self.switch_margin = switch_margin
        self.max_age_s = max_age_s
        self.last_seq = -1
        self.records = []
        self.decisions = []

    def update(self, seq, now, decisions, map_id, voxel_size):
        if seq <= self.last_seq:
            return self.decisions
        consecutive = seq == self.last_seq + 1
        self.last_seq = seq
        previous = [r for r in self.records if now-r['time'] <= self.max_age_s]
        records, output, matched = [], [], set()
        for item in decisions:
            result = dict(item, display_owner=None, display_provisional=False,
                          display_confidence=0., display_reason='insufficient_support')
            output.append(result)
            if 'centroid' not in item or 'thumbnail' not in item or item.get('reason') == 'appearance_conflict':
                continue
            votes = {}
            for sid, count in item.get('votes', {}).items():
                owner = map_id(int(sid))
                if owner is not None:
                    votes[owner] = votes.get(owner, 0) + count
            total = max(1, sum(votes.values()))
            samples = max(1, len(item.get('core_indices', ())) or item.get('samples', 0))
            ranked = sorted((s for s in votes if votes[s] >= 3 and votes[s]/samples >= .25),
                            key=lambda s: (-votes[s], s))
            if not ranked:
                continue
            owner = ranked[0]
            matches = []
            for record in previous:
                if record['owner'] not in votes or votes[record['owner']]/total < .10:
                    continue
                if np.linalg.norm(item['centroid']-record['centroid']) > max(.08, 1.5*voxel_size):
                    continue
                union = np.count_nonzero(item['thumbnail'] | record['thumbnail'])
                iou = np.count_nonzero(item['thumbnail'] & record['thumbnail'])/max(1, union)
                if iou >= .45:
                    matches.append((iou, record))
            matches.sort(key=lambda pair: -pair[0])
            old = matches[0][1] if matches and (len(matches) == 1 or matches[0][0]-matches[1][0] >= .15 or
                all(r['owner'] == matches[0][1]['owner'] for iou,r in matches if matches[0][0]-iou < .15)) else None
            if old is not None:
                matched.update(id(r) for iou,r in matches if r['owner'] == old['owner'])
            challenger, hits = None, 0
            if old is not None and owner != old['owner']:
                advantage = (votes[owner]-votes[old['owner']])/total
                if advantage > self.switch_margin:
                    challenger = owner
                    hits = old['hits']+1 if consecutive and old['challenger'] == owner else 1
                if hits < self.switch_observations:
                    owner = old['owner']
            consensus = map_id(item['winner']) if item.get('winner') is not None else None
            all_same_entity = (len(votes) == 1 and item.get('reason') == 'multiple_supported_owners'
                               and all(map_id(int(s)) is not None for s in item.get('votes', {})))
            provisional = not (owner == consensus or all_same_entity)
            provisional |= len(votes) > 1 and votes[owner]/total < .75
            result.update(display_owner=int(owner), display_provisional=bool(provisional),
                          display_confidence=round(votes[owner]/samples, 3),
                          display_reason='provisional_dominant' if provisional else 'consensus')
            records.append(dict(owner=owner, centroid=item['centroid'].copy(),
                                thumbnail=item['thumbnail'].copy(), time=now,
                                challenger=challenger, hits=hits))
        # Preserve bounded matching history through a missing mask, never stale
        # pixels or a consecutive confirmation run. Current votes are mandatory.
        retained = [dict(r, challenger=None, hits=0) for r in previous if id(r) not in matched]
        self.records = (records + retained)[:512]
        self.decisions = output
        return output
