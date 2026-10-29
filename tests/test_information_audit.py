import copy
import gzip
import json
from pathlib import Path
import unittest

import numpy as np

from patrol_rsi.information_audit import (PublicSignals, SOURCE_NAMES, analytic_continuations,
    current_legal_geometry, pack_geometry, reference_signals, scalar_candidate, score_candidates)
from patrol_rsi.public_stopping import StopConfig, public_frontier


class InformationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'research/2026-10-29-m12c1b32-formal/evidence/explanatory-cases/development-positive.json.gz'
        cls.record = json.loads(gzip.decompress(path.read_bytes()))
        cls.bundle = cls.record['selected_public_snapshots'][0]
        cls.config = StopConfig(**cls.record['formal_identity']['config'])

    def test_source_fusion_ties_and_raw_prior_are_distinguished(self):
        bundle = copy.deepcopy(self.bundle)
        row = bundle['tasks'][0]
        for key, value in [('history_assimilated', 70.), ('nominal', 70.), ('temperature_measurement', 69.), ('cvae_raw', 100.)]:
            row[key].update(available=True, value=value, time_s=750., age_s=0., context_cutoff_s=750.)
        row['velocity_c_s'] = 0.
        signal, source = PublicSignals(bundle).at(np.asarray([0]), 120.)
        self.assertEqual(signal.tolist(), [.25])
        self.assertEqual(SOURCE_NAMES[source[0]], 'nominal')
        self.assertEqual(PublicSignals(bundle).thermal_components(np.asarray([0]), 120.)['cvae_raw'].tolist(), [100.])

    def test_all_current_legal_targets_survive_zero_signal_and_alarm(self):
        bundle = copy.deepcopy(self.bundle)
        row = bundle['tasks'][148]
        row['flow']['value'] = 36.
        row['already_alarm'] = True
        geometry = pack_geometry(current_legal_geometry(bundle))
        i = np.flatnonzero((geometry['agent'] == 14) & (geometry['task'] == 148))
        self.assertEqual(len(i), 1)
        values = score_candidates(PublicSignals(bundle), geometry, self.config)
        self.assertFalse(values['positive_signal_eligible'][i[0]])

    def test_current_candidate_universe_matches_the_original_positive_frontier(self):
        geometry = pack_geometry(current_legal_geometry(self.bundle))
        values = score_candidates(PublicSignals(self.bundle), geometry, self.config)
        old = [v for v in public_frontier(self.bundle, self.config) if v['dispatch_time_s'] == 750.]
        new = [scalar_candidate(self.bundle, geometry, int(i), self.config)
               for i in values['geometry_index'][values['positive_signal_eligible']]]
        self.assertEqual([(v['agent'], v['task']) for v in old], [(v['agent'], v['task']) for v in new])
        for before, after in zip(old, new):
            self.assertEqual(before, after)
        for i in range(len(geometry['task'])):
            point = scalar_candidate(self.bundle, geometry, i, self.config)
            if point['net_score'] is not None:
                self.assertAlmostEqual(values['net_score'][i], point['net_score'], places=11)

    def test_saved_actual_check_score_is_reconstructed_without_truth(self):
        geometry = pack_geometry(current_legal_geometry(self.bundle))
        i = int(np.flatnonzero((geometry['agent'] == 14) & (geometry['task'] == 148))[0])
        point = scalar_candidate(self.bundle, geometry, i, self.config)
        saved = next(v['selected'] for v in self.record['stopping_decisions'] if v.get('selected'))
        self.assertEqual(point, saved)

    def test_professional_reference_uses_flow_and_sensor_error(self):
        arrays = {'rack_temperature_c': np.full((360, 128), 50.),
                  'contact_temperature_c': np.full((360, 16), 50.),
                  'rack_flow_kg_s': np.full((360, 128), 4.5),
                  'public_sensor_temperature_c': np.full((360, 128), 50.)}
        arrays['rack_flow_kg_s'][:, :8] = 2.1
        arrays['public_sensor_temperature_c'][:, :8] = 52.
        signal = reference_signals(arrays)
        self.assertAlmostEqual(signal[0, 144], .5)
        self.assertEqual(signal[0, 160], 1.)
        self.assertEqual(signal[0, 0], 0.)
        self.assertEqual(signal[0, 128], 0.)

    def test_analytic_schedule_accounts_for_lost_and_earlier_confirmation(self):
        tour = {'inspection_events': [{'task': 0, 'agent': 0, 'time_s': 100.}],
                'anomaly_intervals': [{'task': 0, 'onset_s': 85., 'end_s': 115., 'weight': 1.,
                                      'detected': True, 'first_confirmation_s': 100., 'delay_or_censor_s': 15.}]}
        geometry = {'agent': np.asarray([0, 0]), 'task': np.asarray([1, 0]),
                    'dispatch_time_s': np.asarray([80., 80.]), 'finish_s': np.asarray([90., 90.]),
                    'shift_s': np.asarray([30., 30.])}
        reference = np.ones((360, 2))
        result = analytic_continuations(tour, geometry, reference)
        self.assertEqual(result['missed_delta'].tolist(), [1, 0])
        self.assertEqual(result['weighted_wait_delta_s'].tolist(), [15., -10.])
        self.assertEqual(result['lost_confirmations'].tolist(), [1, 0])
        self.assertEqual(result['helpful'].tolist(), [False, True])


if __name__ == '__main__':
    unittest.main()
