import time, os

import pandas as pd
import numpy as np

from ceam import config

from ceam_inputs.auxiliary_files import open_auxiliary_file

from ceam.framework.event import listens_for, emits
from ceam.framework.values import modifies_value, produces_value
from ceam.framework.population import uses_columns

class SimpleIntervention:
    intervention_group = 'age >= 25 and alive == True'

    def setup(self, builder):
        self.reset()
        self.year = 1990 # TODO: better plumbing for this information

    @listens_for('time_step')
    @uses_columns(['age', 'alive'], intervention_group)
    def track_cost(self, event):
        self.year = event.time.year
        if event.time.year >= 1995:
            time_step = config.simulation_parameters.time_step
            self.cumulative_cost += 2.0 * len(event.index) * (time_step / 365.0) # FIXME: charge full price once per year?

    @modifies_value('mortality_rate')
    @uses_columns(['age'], intervention_group)
    def mortality_rates(self, index, rates, population_view):
        if self.year >= 1995:
            pop = population_view.get(index)
            rates.loc[pop.index] *= 0.5
        return rates

    def reset(self):
        self.cumulative_cost = 0

    @listens_for('simulation_end', priority=0)
    def dump_metrics(self, event):
        print('Cost:', self.cumulative_cost)

class SimpleMortality:
    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')

    @produces_value('mortality_rate')
    def base_mortality_rate(self, index):
        return pd.Series(0.01, index=index)

    @listens_for('time_step')
    @emits('deaths')
    @uses_columns(['alive'], 'alive == True')
    def handler(self, event, death_emitter):
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1-np.exp(-effective_rate)
        draw = np.random.random(size=len(event.index))
        affected_simulants = draw < effective_probability
        event.population_view.update(pd.Series(False, index=event.index[affected_simulants]))
        death_emitter(event.split(affected_simulants.index))

class SimpleMetrics:
    def setup(self, builder):
        self.reset()
        with open_auxiliary_file('Life Table') as f:
            self.life_table = builder.lookup(pd.read_csv(f), key_columns=(), parameter_columns=('age',))

    @listens_for('deaths')
    def count_deaths_and_ylls(self, event):
        self.deaths += len(event.index)

        t = self.life_table(event.index)
        self.ylls += t.sum()

    def reset(self):
        self.start_time = time.time()
        self.deaths = 0
        self.ylls = 0

    def run_time(self):
        return time.time() - self.start_time

    @listens_for('simulation_end', priority=0)
    def dump_metrics(self, event):
        print('\nWith intervention:')
        print('Deaths:', self.deaths)
        print('YLLs:', self.ylls)
        print('Run time:', self.run_time())
