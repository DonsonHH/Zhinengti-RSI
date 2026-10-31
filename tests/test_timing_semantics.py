"""Historical professional semantics and unchanged discovery accounting."""
import unittest
from types import SimpleNamespace
import numpy as np

from patrol_rsi.facility import FacilityConfig
from patrol_rsi.facility_observation import FacilityObserver,InspectionTask,inspection_tasks
from patrol_rsi.active_inspection import measured_anomaly
from patrol_rsi.inspection_metrics import evaluate_episode
from patrol_rsi.patrol_timing import gate_candidate


def legacy_row(kind,signal,alarm):
    return {'agent':0,'task':128 if kind=='electrical' else 160,'baseline_next':129 if kind=='electrical' else 161,
        'kind':kind,'signal_name':'contact_heat' if kind=='electrical' else 'verified_sensor_bias','signal':signal,
        'threshold':.05 if kind=='electrical' else .5,'available':True,'measurement_age_s':0.,'signal_source':'qualified_reference',
        'observed_value':None,'observed_unit':'degC','estimate_age_s':None,'weight_scaled':.75,'age_ratio':.5,'eta_ratio':.1,
        'candidate_finish_s':130.,'tour_target_finish_s':230.,'lead_s':100.,'shift_s':10.,'already_alarm':alarm}


class ProfessionalSemanticsTests(unittest.TestCase):
    def test_electrical_qualified_positive_and_alarm_share_the_same_reference(self):
        observer=FacilityObserver(FacilityConfig(racks=8));task=InspectionTask('electrical:0','electrical',0,'electrical',20.,(0.,0.))
        for time,value in enumerate((38.,60.,60.1,61.5,80.,38.),1):
            measurement={'pod_index':0,'contact_temperature_c':value}
            observer.complete_inspection(task,'electrical',measurement,time*5,True)
            alert=measured_anomaly(task,measurement);signal=max(observer.last_contact[0]-60,0)/30
            self.assertFalse(gate_candidate(legacy_row('electrical',float(signal),bool(alert))))
            if signal>=.05:self.assertTrue(alert)
    def test_control_confirmation_and_recovery_update_the_flag_and_alarm_together(self):
        observer=FacilityObserver(FacilityConfig(racks=8));task=InspectionTask('control:0','control',0,'control',15.,(0.,0.))
        for time,error in enumerate((0.,-1.,-3.,1.,3.,0.),1):
            measurement={'pod_index':0,'controller_sensor_error_c':error}
            observer.complete_inspection(task,'control',measurement,time*5,True)
            alert=measured_anomaly(task,measurement);verified=bool(observer.sensor_bias_verified[0])
            self.assertEqual(verified,alert)
            self.assertFalse(gate_candidate(legacy_row('control',float(verified),bool(alert))))
        self.assertFalse(observer.sensor_bias_verified[0])
    def test_recheck_of_one_known_anomaly_does_not_create_another_discovery(self):
        config=FacilityConfig(racks=8);tasks=inspection_tasks(config)
        arrays={'rack_temperature_c':np.full((360,8),66.),'contact_temperature_c':np.full((360,1),38.),
            'rack_flow_kg_s':np.full((360,8),5.),'public_sensor_temperature_c':np.full((360,8),66.)}
        roles=('thermal','electrical','cooling','control')
        scalar={'overdue_fraction':0.,'maximum_age_s':200.,'per_role_overdue':dict.fromkeys(roles,0.),'per_role_maximum_age_s':dict.fromkeys(roles,200.),'public_reward':0.}
        env=SimpleNamespace(config=config,tasks=tasks,trace=SimpleNamespace(arrays=arrays,metadata={'unserved_IT_kwh':0.}),
            asset_weights=np.ones(len(tasks)),ever=np.zeros(len(tasks),dtype=bool),scalar_trace=[scalar],inference_seconds=0.,agents=[])
        env.ever[0]=True;first={'task':0,'time_s':100.,'alarm':True};recheck={'task':0,'time_s':200.,'alarm':True}
        env.events=[first];before=evaluate_episode(env)[0]
        env.events=[first,recheck];after=evaluate_episode(env)[0]
        self.assertEqual(before['confirmed_segments'],1);self.assertEqual(after['confirmed_segments'],1)
        self.assertEqual(before['missed_segments'],after['missed_segments'])
        self.assertEqual(before['consequence_weighted_censored_delay_s'],after['consequence_weighted_censored_delay_s'])


if __name__=='__main__':unittest.main()
