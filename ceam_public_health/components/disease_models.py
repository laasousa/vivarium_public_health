# ~/ceam/ceam/modules/disease_models.py

from datetime import timedelta

from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, RateTransition, ProportionTransition
from ceam_inputs import get_incidence
from ceam_inputs.gbd_ms_functions import get_post_mi_heart_failure_proportion_draws, get_angina_proportions, get_asympt_ihd_proportions, load_data_from_cache, get_disability_weight
from ceam_inputs.gbd_ms_auxiliary_functions import normalize_for_simulation


def heart_disease_factory():
    module = DiseaseModel('ihd')

    healthy = State('healthy', key='ihd')

    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    year_end = config.getint('simulation_parameters', 'year_end')

    # Calculate an adjusted disability weight for the acute heart attack phase that
    # accounts for the fact that our timestep is longer than the phase length
    # TODO: This doesn't account for the fact that our timestep is longer than 28 days
    timestep = config.getfloat('simulation_parameters', 'time_step')
    weight = 0.43*(2/timestep) + 0.07*(28/timestep)
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=weight, dwell_time=timedelta(days=28), modelable_entity_id=1814, prevalence_meid=1814)

    #
    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1821), modelable_entity_id=2412, prevalence_meid=1821)
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1822), modelable_entity_id=2412, prevalence_meid=1822)
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1823), modelable_entity_id=2412, prevalence_meid=1823)

    asymptomatic_angina = ExcessMortalityState('asymptomatic_angina', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1823), modelable_entity_id=1817, prevalence_meid=3102)
    mild_angina = ExcessMortalityState('mild_angina', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1818), modelable_entity_id=1817, prevalence_meid=1818)
    moderate_angina = ExcessMortalityState('moderate_angina', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1819), modelable_entity_id=1817, prevalence_meid=1819)
    severe_angina = ExcessMortalityState('severe_angina', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=1820), modelable_entity_id=1817, prevalence_meid=1820)

    asymptomatic_ihd = ExcessMortalityState('asymptomatic_ihd', disability_weight=get_disability_weight(dis_weight_modelable_entity_id=3233), modelable_entity_id=3233, prevalence_meid=3233)

    heart_attack_transition = RateTransition(heart_attack, 'incidence_rate.heart_attack', get_incidence(1814))
    healthy.transition_set.append(heart_attack_transition)

    heart_failure_buckets = TransitionSet(allow_null_transition=False, key="heart_failure_split")
    heart_failure_buckets.extend([
        ProportionTransition(mild_heart_failure, proportion=0.182074),
        ProportionTransition(moderate_heart_failure, proportion=0.149771),
        ProportionTransition(severe_heart_failure, proportion=0.402838),
        ])

    angina_buckets = TransitionSet(allow_null_transition=False, key="angina_split")
    angina_buckets.extend([
        ProportionTransition(asymptomatic_angina, proportion=0.304553),
        ProportionTransition(mild_angina, proportion=0.239594),
        ProportionTransition(moderate_angina, proportion=0.126273),
        ProportionTransition(severe_angina, proportion=0.32958),
        ])
    healthy.transition_set.append(RateTransition(angina_buckets, 'incidence_rate.non_mi_angina', get_incidence(1817)))

    heart_attack.transition_set.allow_null_transition=False

    # TODO: Need to figure out best way to implemnet functions here
    # TODO: Need to figure out where transition from rates to probabilities needs to happen
    hf_prop_df = load_data_from_cache(get_post_mi_heart_failure_proportion_draws, col_name='proportion', src_column='draw_{draw}', location_id=location_id, year_start=year_start, year_end=year_end)
    angina_prop_df = load_data_from_cache(get_angina_proportions, col_name='proportion', src_column='angina_prop_{draw}')#, year_start=year_start, year_end=year_end)
    asympt_prop_df = load_data_from_cache(get_asympt_ihd_proportions, col_name='proportion', src_column='asympt_prop_{draw}', location_id=location_id)#, year_start=year_start, year_end=year_end)

    # post-mi transitions
    # TODO: Figure out if we can pass in me_id here to get incidence for the correct cause of heart failure
    # TODO: Figure out how to make asymptomatic ihd be equal to whatever is left after people get heart failure and angina
    heart_attack.transition_set.append(ProportionTransition(heart_failure_buckets, proportion=hf_prop_df))
    heart_attack.transition_set.append(ProportionTransition(angina_buckets, proportion=angina_prop_df))
    heart_attack.transition_set.append(ProportionTransition(asymptomatic_ihd, proportion=asympt_prop_df))

    mild_heart_failure.transition_set.append(heart_attack_transition)
    moderate_heart_failure.transition_set.append(heart_attack_transition)
    severe_heart_failure.transition_set.append(heart_attack_transition)
    asymptomatic_angina.transition_set.append(heart_attack_transition)
    mild_angina.transition_set.append(heart_attack_transition)
    moderate_angina.transition_set.append(heart_attack_transition)
    severe_angina.transition_set.append(heart_attack_transition)
    asymptomatic_ihd.transition_set.append(heart_attack_transition)

    module.states.extend([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, asymptomatic_angina, mild_angina, moderate_angina, severe_angina, asymptomatic_ihd])
    return module


def stroke_factory():
    module = DiseaseModel('hemorrhagic_stroke')

    healthy = State('healthy', key='hemorrhagic_stroke')
    # TODO: need to model severity splits for stroke. then we can bring in correct disability weights (dis weights
    # correspond to healthstate ids which correspond to sequela) 
    hemorrhagic_stroke = ExcessMortalityState('hemorrhagic_stroke', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=9311)
    ischemic_stroke = ExcessMortalityState('ischemic_stroke', disability_weight=0.32, dwell_time=timedelta(days=28), modelable_entity_id=9310)
    chronic_stroke = ExcessMortalityState('chronic_stroke', disability_weight=0.32, modelable_entity_id=9312)

    hemorrhagic_transition = RateTransition(hemorrhagic_stroke, 'incidence_rate.hemorrhagic_stroke', get_incidence(9311))
    ischemic_transition = RateTransition(ischemic_stroke, 'incidence_rate.ischemic_stroke', get_incidence(9310))
    healthy.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    hemorrhagic_stroke.transition_set.append(Transition(chronic_stroke))
    ischemic_stroke.transition_set.append(Transition(chronic_stroke))

    chronic_stroke.transition_set.extend([hemorrhagic_transition, ischemic_transition])

    module.states.extend([healthy, hemorrhagic_stroke, ischemic_stroke, chronic_stroke])

    return module


# End.