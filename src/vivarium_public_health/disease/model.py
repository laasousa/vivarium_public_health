import numbers

import pandas as pd
import numpy as np

from vivarium import VivariumError
from vivarium.framework.state_machine import Machine

from vivarium_public_health.disease import (SusceptibleState, ExcessMortalityState, TransientDiseaseState,
                                        RateTransition, ProportionTransition)


class DiseaseModelError(VivariumError):
    pass


class DiseaseModel(Machine):
    def __init__(self, cause, initial_state=None, get_data_functions=None, cause_type="cause", **kwargs):
        self.cause = cause
        self.cause_type = cause_type
        super().__init__(cause, **kwargs)

        if initial_state is not None:
            self.initial_state = initial_state.state_id
        else:
            self.initial_state = self._get_default_initial_state()

        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

        if 'csmr' not in self._get_data_functions:
            self._get_data_functions['csmr'] = lambda cause, builder: builder.data.load(
                f"{self.cause_type}.{cause}.cause_specific_mortality")

    @property
    def condition(self):
        return self.state_column

    def setup(self, builder):
        super().setup(builder)

        self._csmr_data = self._get_data_functions['csmr'](self.cause, builder)
        self.config = builder.configuration
        self._interpolation_order = builder.configuration.interpolation.order

        builder.value.register_value_modifier('csmr_data', modifier=self.get_csmr)
        builder.value.register_value_modifier('epidemiological_point_measures', modifier=self.prevalence)
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

        self.population_view = builder.population.get_view(['age', 'sex', self.condition])
        builder.population.initializes_simulants(self.load_population_columns,
                                                 creates_columns=[self.condition],
                                                 requires_columns=['age', 'sex'])
        self.randomness = builder.randomness.get_stream('{}_initial_states'.format(self.condition))

        builder.event.register_listener('time_step', self.time_step_handler)
        builder.event.register_listener('time_step__cleanup', self.time_step__cleanup_handler)

    def _get_default_initial_state(self):
        susceptible_states = [s for s in self.states if isinstance(s, SusceptibleState)]
        if len(susceptible_states) != 1:
            raise DiseaseModelError("Disease model must have exactly one SusceptibleState.")
        return susceptible_states[0].state_id

    def time_step_handler(self, event):
        self.transition(event.index, event.time)

    def time_step__cleanup_handler(self, event):
        self.cleanup(event.index, event.time)

    def get_csmr(self):
        return self._csmr_data

    @staticmethod
    def assign_initial_status_to_simulants(simulants_df, states, initial_state, randomness):
        simulants = simulants_df[['age', 'sex']].copy()
        sequelae, weights = zip(*states.items())
        sequelae += (initial_state,)
        for w in weights:
            w.reset_index(inplace=True, drop=True)
        weights += ((1 - np.sum(weights, axis=0)),)
        simulants.loc[:, 'condition_state'] = randomness.choice(simulants.index, sequelae,
                                                                np.array(weights).T)
        return simulants

    def load_population_columns(self, pop_data):
        population = self.population_view.get(pop_data.index, omit_missing_columns=True)

        assert self.initial_state in {s.state_id for s in self.states}

        state_map = {s.state_id: s.prevalence_data(pop_data.index) for s in self.states
                     if hasattr(s, 'prevalence_data') and s.prevalence_data is not None}

        if state_map and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population['sex_id'] = population.sex.apply({'Male': 1, 'Female': 2}.get)

            condition_column = self.assign_initial_status_to_simulants(population, state_map,
                                                                       self.initial_state, self.randomness)

            condition_column = condition_column.rename(columns={'condition_state': self.condition})
        else:
            condition_column = pd.Series(self.initial_state, index=population.index, name=self.condition)
        self.population_view.update(condition_column)

    def prevalence(self, index, age_groups, sexes, all_locations, duration, cube):
        root_location = self.config.input_data.location
        pop = self.population_view.manager.population.ix[index].query("alive == 'alive'")
        causes = set(pop[self.condition]) - {state.state_id for state in self.states
                                             if isinstance(state, SusceptibleState)}
        if all_locations:
            locations = set(pop.location) | {-1}
        else:
            locations = {-1}
        for low, high in age_groups:
            for sex in sexes:
                for cause in causes:
                    for location in locations:
                        sub_pop = pop.query('age >= @low and age < @high and sex == @sex')
                        if location >= 0:
                            sub_pop = sub_pop.query('location == @location')
                        if not sub_pop.empty:
                            affected = (sub_pop[self.condition] == cause).sum()
                            cube = cube.append(pd.DataFrame({'measure': 'prevalence',
                                                             'age_low': low,
                                                             'age_high': high,
                                                             'sex': sex,
                                                             'location': location if location >= 0 else root_location,
                                                             'cause': cause, 'value': affected/len(sub_pop),
                                                             'sample_size': len(sub_pop)},
                                                            index=[0]).set_index(
                                ['measure', 'age_low', 'age_high', 'sex', 'location', 'cause']))
        return cube

    def to_dot(self):
        """Produces a ball and stick graph of this state machine.

        Returns
        -------
        `graphviz.Digraph`
            A ball and stick visualization of this state machine.
        """
        from graphviz import Digraph
        dot = Digraph(format='png')
        for state in self.states:
            if isinstance(state, ExcessMortalityState):
                dot.node(state.state_id, color='red')
            elif isinstance(state, TransientDiseaseState):
                dot.node(state.state_id, style='dashed', color='orange')
            elif isinstance(state, SusceptibleState):
                dot.node(state.state_id, color='green')
            else:
                dot.node(state.state_id, color='orange')
            for transition in state.transition_set:
                if transition._active_index is not None:  # Transition is a triggered transition
                    dot.attr('edge', style='bold')
                else:
                    dot.attr('edge', style='plain')

                if isinstance(transition, RateTransition):
                    dot.edge(state.state_id, transition.output_state.state_id, transition.label(), color='blue')
                elif isinstance(transition, ProportionTransition):
                    dot.edge(state.state_id, transition.output_state.state_id, transition.label(), color='purple')
                else:
                    dot.edge(state.state_id, transition.output_state.state_id, transition.label(), color='black')

            if state.transition_set.allow_null_transition:
                if hasattr(state, '_dwell_time'):
                    if isinstance(state._dwell_time, numbers.Number):
                        if state._dwell_time != 0:
                            label = "dwell_time: {}".format(state._dwell_time)
                            dot.edge(state.state_id, state.state_id, label, style='dotted')
                        else:
                            dot.edge(state.state_id, state.state_id, style='plain')
                    else:
                        dot.edge(state.state_id, state.state_id, style='dotted')
        return dot

    def metrics(self, index, metrics):
        population = self.population_view.get(index, query="alive == 'alive'")
        metrics[self.condition + '_count'] = (population[self.condition] != 'susceptible_to_' + self.condition).sum()
        return metrics