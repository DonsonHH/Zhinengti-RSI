"""Future independent E2 designs only; this script cannot launch experiments."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research/2026-10-31-m12e-timing-schema'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    e1=json.loads((STAGE/'E1-Timing-v1.json').read_bytes())
    task={'protocol':'M12E2-Task-Paired-v1','status':'future_design_for_independent_review_not_executed','baseline':'6174f2ab3ae227b8bec6f83b05fdca512bb84192',
        'question':'Does the fixed professional timing gate plus frozen PPO improve independent patrol outcomes beyond Tour, human and nonlearning gated ranking?',
        'launch_requirements':['M12E-E1 opportunity gate PASS','M12E-E1 timing gate PASS','Independent exact-source/schema/model/resource review','Actual regression and immutable data manifest and final gate implementation'],
        'arms':e1['arms'],'method_definitions':'Exactly reviewed E1 source/candidate gate/rankers, no newcandidate or parameter from E1 result; manual gate not credited to learning or RSI.',
        'E1_protocol_SHA256':sha(STAGE/'E1-Timing-v1.json'),'fixed_manifest_SHA256':sha(STAGE/'process-manifest.json'),
        'partitions':{'development_already_used':[33000,33001],'validation':[33020,33021,33022,33023],'final_test':[33040,33041,33042,33043]},
        'units':'9uniquephysicalprocesses/seed x2pairednoise x5methods;4independentseedgroups perconfirmationstage,not72independentenvironments.',
        'primary':'evidence_gate_PPO versus Tour AND human_professional AND evidence_gate_urgency; unconditional_PPO secondary timing attribution.',
        'aggregation':'Withincondition originalconsequenceweighted delay/censor;meanpairednoise withinprocess;equal event-bearingprocesses forwait andall9formiss withinseed;equalseeds. Healthy cost/safety/miss retained,waitnull.',
        'validation_GO':'All safety/source/resource checks; perseed no increased miss versus each3primary baselines; versus EACH baseline lower macro miss ORsame macro miss and>=5%lower macro wait.',
        'final_open':'Only after all5programs/models/sourceSHA andvalidationGO sealed. No failure masking byTour, no extra candidate. Failure stops whole taskcampaign with final sources ungenerated.',
        'final_confirmation':'Score all5locked methods once;independently replay each primary episode and audit qualification/route/measurement/oneCHECK/fullreturn/head-tailrevisit. Report allseed/process/noise paired values, tooluseandcost. PointGOdoesnotprovepopulation effect.',
        'statistics':'Report4seed differences; any2000seedbootstrap exploratory descriptive only; no event/noise pseudo-replication, no positive-publication claim from singlemacro.',
        'budgets':{'validation_primary_episodes':360,'final_primary_episodes':360,'primary_episode_cap':720,'independent_replay_cap':720,'all_environment_execution_cap':1440,
            'GPU_allocations':0,'LLM_calls':0,'CPU_minutes_cap':960,'max_parallel_CPU_jobs':2,'CPU_per_job':2,'RAM_GB':8,'storage_cap_bytes':3000000000},
        'current_stage_permissions':'Design only. M12-E drivers implementdevelopment exclusively; no currentvalidation/final launch.',
        'prohibited':['oldM12Dtestreuse','thresholdrelaxation','newtraining','extraworkforceoractionspace','oracleonline','postfinalreselection']}
    write(STAGE/'E2-Task-Paired-v1.json',task)
    meta={'protocol':'M12E2-Legal-Meta-v1','status':'future_design_for_independent_review_not_executed','question':'After reliable legal proposal and a confirmed task opportunity, does a changed improver improve fresh-task search at matched3proposal slots and full preparation cost?',
        'launch_requirements':['E0 exactfrozen format gate PASS','E1 opportunityANDtiming gates PASS','Independent task-level validation fromE2Task PASS','New adapter/schema/executiongate independentlyreviewed andregressed','No currentstage permission toexecute'],
        'model':{'id':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218','thinking':False,'precision':'float16','temperature':.7,'top_p':.8,'top_k':20,
            'input_cap_per_call':4096,'output_cap_per_call':768,'weights_updated':False},
        'proposal_interface':'Use full_nested if its E0gatepassed; else compact_leaves onlyifpassed, explicitdeterministicassemblylabel. No switch aftermetaoutcomes; no freeformatrepair or retry.',
        'meta_update_space':'Unchanged ControlledImprover:1..3distinctdiagnostics,3fixedinstructiontemplates,memory/generator/dedup/acceptance/failureenum. No task/evaluator/budget/physicalcoreedit.',
        'NEW_task_adapter_semantics':{'physical_core':'Same18staff/qualified ownedtasks/fullreturn/effectivemeasurement/revisit/oneglobalCHECK.',
            'task_syntax':'ExistingPatrolProgramsubset:refresh_s30,cooldown_s60,fallbackTourorPPO,<=3typedorderedrules inoriginal signal/threshold/action domains; rejectotherintervals, do notrepair.',
            'gate':'SamefixedE1 professional eligibility, positivelead andsignal*weight*lead>.05shift; no rule can bypassgate. All learned/randomtaskprograms share gate; Tour always legal.',
            'rule_semantics':'Firstmatchingrule overgate-eligibleoptionalset; return_tour preservesquota; signal/urgency/PPO rank eligible subset using originalrank definitions. No matchingrule: Tour fallbackpreserves; PPO fallbackrankseligible only. AfterglobalCHECK all returnTour.',
            'disclosure':'New task interpreter semantics, manually designed and fixed acrosssearchgroups; not attributedtoQwen or meta discovery. Must be implemented and separatelyreviewed before any future calls.'},
        'groups':{'human_fixed':'ExactoriginalE1 human_professional reference,0candidate/modelcalls. No source/qualification permissions differ.',
            'random_search':{'task_candidate_slots':8,'model_calls':0},'fixed_Qwen':{'task_candidate_slots':8,'model_call_cap':8,'prefix3_saved':True},
            'controlled_meta':{'meta_protocol_slots':1,'paired_old_candidates':2,'paired_new_candidates':2,'fresh_task_candidates':3,'model_call_cap':8}},
        'logical_evaluation_references':['Tour','human_professional','evidence_gate_urgency','random8','fixed3','fixed8','controlled_meta3'],
        'generation_repeats':[9601,9602,9603],
        'partitions':{'meta_development':[34000,34001],'fresh_task_adapt':[34100,34101],'validation':[34120,34121],'final_test':[34140,34141,34142,34143]},
        'fresh_phases':'NewM9manifestwithseparatenamespace andallseed/numericfingerprints disjointfromM12D/E1/E2Task;9processes/seed2noise. Must freezeexactonsetsbefore anygeneration, no oldfixedphasewaitrules.',
        'paired_meta':'Defaultprotocolincumbent; oneproposedlegalprotocol. Old/new fromsameTour,emptyhistory/archive/cache,matchedmodelseeds,twoattemptseach. Strictannotateddevelopment improvement+safetyrequiredtoretain; invalid/duplicate/failure consumes. Onlyretainedprotocoltransfers.',
        'transfer':'FreshprogramTour,emptyexperience/archive/validationresults/cache; onlyschema-validatedprotocol. No taskIDs, seeds, phase, evaluatorlabels inprompts; onlywhitelistedpublicdiagnostics andaccept/rejectfeedback.',
        'comparisons':{'primary':'fixedprefix3 vsmeta fresh3 onnewtask, all5meta-prepcalls/tokens/env/GPU costs separately included.',
            'secondary':'fixed8 versusmeta8 totalcallcaps; directsearchhas8fresh opportunities whilemeta has3, report unequal downstreamopportunity. Random8 andhuman0 costreference, notequalcompute claim.'},
        'budgets_per_repeat':{'model_call_cap':16,'task_candidate_attempts':23,'meta_protocol_attempts':1,'meta_environment_episode_cap':180,
            'fresh_adapt_environment_episode_cap':792,'validation_environment_episode_cap':252,'final_environment_episode_cap':504,'environment_episode_cap':1728},
        'all3_caps':{'model_calls':48,'task_candidate_attempts':69,'meta_protocol_attempts':3,'environment_episodes':5184,'input_tokens':196608,'output_tokens':36864,
            'GPU_allocations':3,'GPU_seconds':10800,'CPU_minutes':2400,'storage_bytes':4000000000},
        'accounting':'Chargeeveryattemptbeforegeneration/evaluation; perpartitionimmutablecache withactual/cachereferencesseparate; duplicatesinvalids/timeoutsnoresampling. Unknowntokens labelledandchargedupperbound. Allprep/inference/env/auditandfailedstartupcosts included.',
        'validation_fallback':'EachnonTourreference: allseedmiss<=Tour and macro fewermisses ORtie strictlylesswait; otherwise actualfinalPG=Tour. Any safety/source/budgetfailure stopsall, no hiddenTourmask. Saveall7nominal/actualmappings beforefinal.',
        'final_open':'All3generationrepeats andall7mappings complete+source/model/resource seal; onceonfixedactualPGs, cacheduplicatesreported. No nominalrescoring afterfinal.',
        'GO':'Taskandmetaeffectsreportedseparately. MetaGO requiresperseednoincreasedmiss vsfixed3 andfewermacro miss OR>=5%waitgain in EACH3generationrepeats in validationANDfinal. Also taskvsTour/human/nonlearning baselines must not regress to calloverallproposaluseful. Fullcostincluded, accepts count not gain.',
        'statistical_unit':'3generationrepeats reuse4finalphysicalseedgroups,not12independentenvironments;3repeat contrasts+4seedpairedmeans required. Independentrawrecords/auditsandmetricreconstruction beforedelivery.',
        'current_stage_execution':False,'no_78_call_search':'Thisfuturedesigncaps48calls; nofuturemeta callsauthorizedorlaunchedinM12E.'}
    write(STAGE/'E2-Legal-Meta-v1.json',meta)
    print('M12E_FUTURE_DESIGNS_ONLY',sha(STAGE/'E2-Task-Paired-v1.json'),sha(STAGE/'E2-Legal-Meta-v1.json'))


if __name__=='__main__':main()
