"""Concrete future protocol, not an execution entry point or dataset generator."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];stage=root/'research/2026-11-01-m12g-local';F=root/'research/2026-10-31-m12f-timing-skill'
    G=read(a.run_root/'G0-complete.json');study=read(stage/'G0-G1a-Study-v1.json')
    history_path=F/'evidence/historical-identities.json';history=read(history_path)
    reserved={v['specification']['seed'] for v in read(F/'cases-v1.json')['future']}
    excluded=set(history['seed_groups'])|reserved|{33000,33001,35020,35021}
    validation=list(range(38020,38024));final=list(range(38040,38044));assert not(set(validation+final)&excluded)
    pool=[v for v in range(95,900,5) if v not in (150,335,520,705)];future=[]
    fixtures=(('healthy',900),('coldplate_fouling',180),('dc_contact_hotspot',180),('sensor_bias',900),('aux_feeder_loss+cdu_fouling',180))
    for phase,seeds in (('validation',validation),('final',final)):
        for seed in seeds:
            for family,duration in fixtures:
                onset=pool[int(hashlib.sha256(f'M12G-G1b-phase-v1|{phase}|{seed}|{family}'.encode()).hexdigest(),16)%len(pool)]
                for noise in (.2,1.):future.append({'phase':phase,'seed':seed,'scenario':family,'duration_s':duration,'fault_start_s':onset,'noise_c':noise,'status':'proposed_not_generated'})
    pre=a.run_root/'G1a-precheck/complete.json';precheck=read(pre) if pre.exists() else None
    ready=bool(G['local_mechanism_supported'] and precheck and precheck['complete'] and precheck['interface_passed'])
    value={'protocol':'M12G-G1b-four-group-DRAFT-v1','status':'DRAFT_READY_FOR_SEPARATE_GOAL_REVIEW' if ready else 'DRAFT_NOT_READY',
        'execution_authorized_by_this_file':False,'G1b_calls_or_data_generated':False,'G2_included':False,
        'readiness':{'G0_local_mechanism_supported':G['local_mechanism_supported'],'G1a_precheck_executed':precheck is not None,
            'G1a_interface_passed':precheck['interface_passed'] if precheck else None,'G0_receipt_SHA256':sha(a.run_root/'G0-complete.json'),
            'precheck_receipt_SHA256':sha(pre) if pre.exists() else None,'reason':'Usable local mechanism plus typed interface needed; not a PPO/all-predictors prerequisite.'},
        'common_parent':'InitialE1 every slot; ownneutralprevious2patches allgroups; no winner-parent leakage to Q0',
        'groups':{'S':'Same1-2edit language; fixed stratifiedoperation/path/value sampler, no model',
            'Q0':'FixedQwen with no run observations or acceptancebits',
            'Q1':'FixedQwen with F-style representative retrieval',
            'Q2':'FixedQwen with decision-difference public feedback package'},
        'proposal_budget':{'repeats':[38101,38102,38103],'slots_per_group':4,'groups':4,'total_slots':48,'actual_LLM_calls_cap':36},
        'feedback_contract':{'access':'Q1/Q2 same initial+ownrecent2completedrun public records and acceptancebits; no hidden scores/events/oracle/seed/scenario in prompts',
            'card_limit':8,'input_token_cap':6144,'output_token_cap':1024,'initial_Q1_Q2_identical':True,
            'Q2':'fourprofessioncoverage then up to4public differences/positiveCHECKobservations/normalcontrasts; shadows not parentcounterfactualoutcomes',
            'mechanism_claim':'Decisiondifferencefeedback package vsF-stylepackage; doesnotisolatepure retrievalordering alone'},
        'strong_references':['Tour','original_human_professional','initial_E1','fixed_information','G0_best_local_reference'],
        'G0_reference':G['selected_local_reference'],'G0_reference_is_not_secret_Q2_parent':True,
        'development':{'known_seeds':[33000,33001],'conditions':20,'multiple_design_reuse_disclosed':True,'selection':'allseedmiss<=initial + strict lexicographic macro miss/wait/CHECK/walk,abs1e-9'},
        'future_manifest':future,'future_identity_scope':{'checked_against':[str(history_path.relative_to(root)),str((F/'cases-v1.json').relative_to(root))],
            'historical_registry_SHA256':sha(history_path),'scope':'Known historical identities and allFreservedcases; recheck global history/reservations at future actualfreeze; no claiming complete global reservation census.'},
        'confirmatory_claims':{'feedback':'Q2strictgainQ1 ineach3repeat +strongreference nonregression',
            'experience':'Q2strictgainQ0 ineach3repeat +strongreference nonregression',
            'task':'Q2strictgainS andALL5strongreferences ineach3repeat'},
        'strict_gain':'Everyseedmissnoincrease AND fewermacroMiss OR sameMiss>=5%relativewaitgain OR samemiss/wait>=5%CHECKreduction&walknoincrease;1e-9tol,zero/undefineddenominatornotgain;cost-onlylabelseparate',
        'final_open':'All3selectedmappings/source/models frozenbeforevalidationgeneration; anyclaimvalidates ->onefinalopen; finalmustrepeatSAMEclaim, no reselection',
        'analysis_unit':'Noise->physicalprocess->seed macro;3modelrepeats reuse4confirmationseeds, not12independentrooms. Seedclustered summaries/uncertainty with smallsample limitations.',
        'environment_caps':{'G1b_development':1060,'validation':680,'final':680,'G1b_total':2420,'prior_G0':480,'combined_G0_G1_cap':2900},
        'resource_planning_envelope':{'combined_G0_G1_CPU_core_minutes':900,'combined_GPU_seconds':7200,'combined_storage_bytes':2000000000,'parallel_allocations':2,
            'status':'Future schedulemustsubtractactualfirstgoalspending andverifyfitsbefore anynewcalls/data; not a newallocation'},
        'mandatory_future_work_before_start':['Freeze stratifiedSsampling andmatchedgenseed schedule; no extra samplinguntilpositive',
            'Bindactualpatch/feedbacksourceandnewregistry checkedmanifest, source/model hashes and resourceledger',
            'Predeclare allactual17policyidentities/cache scope andsameclaim validation/final gate',
            'Validate interface treatment controls and pilot eventcoverage scope without openingfinal']}
    with a.output.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print('G1B_DRAFT_ONLY',value['status'],len(future),'noactualcalls/data',flush=True)


if __name__=='__main__':main()
