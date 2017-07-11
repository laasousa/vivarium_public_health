from datetime import timedelta

import pandas as pd

from vivarium import config

from vivarium.framework.util import rate_to_probability

from ceam_inputs import get_age_specific_fertility_rates

from vivarium.framework.state_machine2 import Transition, TransitionSet, record_event_time, active_after_delay, new_state_side_effect
from vivarium.framework.event import listens_for
from vivarium.framework.values import produces_value
from vivarium.framework.population import uses_columns, creates_simulants


class Fertility:
    def setup(self, builder):
        time_step = config.simulation_parameters.time_step
        time_step = timedelta(days=time_step)

        self.transitions = TransitionSet('fertility')
        conception = Transition(probability=self.conception_probability,
                                side_effect_functions=[
                                    record_event_time('last_conception_time', builder.clock()),
                                    new_state_side_effect('pregnancy', 'pregnant')
                                ]
                     )
        birth      = Transition(probability=1,
                                side_effect_functions=[
                                    record_event_time('last_birth_time', builder.clock()),
                                    new_state_side_effect('pregnancy', 'not_pregnant'),
                                    self.add_child
                                ],
                                activation_functions=[
                                    active_after_delay(timedelta(days=9*30.5), 'last_conception_time', builder.clock())
                                ]
                     )

        self.transitions.table['not_pregnant'].append(conception)
        self.transitions.table['not_pregnant'].append(Transition.RESIDUAL)
        self.transitions.table['pregnant'].append(birth)
        self.transitions.table['pregnant'].append(Transition.RESIDUAL)

        self.conception_rate = builder.rate('fertility.conception_rate')
        self.randomness = builder.randomness('initial_fertility')

        # TODO: Use the uncertainty bounds that this table has for some but not all locations
        self.asfr = builder.lookup(get_age_specific_fertility_rates()[['year', 'age', 'mean_value']], key_columns=(), parameter_columns=('year', 'age',))

        return [self.transitions]

    @listens_for('time_step')
    @uses_columns(['pregnancy'], 'alive == "alive" and sex == "Female"')
    def step(self, event):
        self.transitions.transition(event.population.pregnancy)

    @produces_value('fertility.conception_rate')
    @uses_columns(['fertility'], 'pregnancy == "not_pregnant"')
    def base_conception_rate(self, index, population_view):
        pop = population_view.get(index)

        return self.asfr(index) * pop.fertility

    @listens_for('initialize_simulants')
    @uses_columns(['sex', 'fertility'])
    def make_fertility_column(self, event):
        fertility = pd.Series(0.0, name='fertility', index=event.index)
        women = event.population.sex == 'Female'
        fertility[women] = self.randomness.get_draw(event.index[women])*2
        event.population_view.update(fertility)

    @listens_for('initialize_simulants')
    @uses_columns(['pregnancy'])
    def make_pregnancy_column(self, event):
        event.population_view.update(pd.Series('not_pregnant', index=event.index))

    def conception_probability(self, index):
        return rate_to_probability(self.conception_rate(index))

    @creates_simulants
    def add_child(self, index, creator):
        if len(index) > 0:
            creator(len(index), population_configuration={'initial_age': 0})
