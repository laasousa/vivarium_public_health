"""
==============
Disease Models
==============

This module contains tools for modeling diseases in multi-state lifetable
simulations.

"""
import numpy as np
import pandas as pd


class Disease:
    """This component characterises a chronic disease.

    It defines the following rates, which may be affected by interventions:

    - `<disease>.incidence`
    - `<disease>.remission`
    - `<disease>.mortality`

    where `<disease>` is the name as provided to the constructor.

    Parameters
    ----------
    name
        The disease name (referred to as `<disease>` here).

    """

    def __init__(self, name):
        self._name = name
        self.configuration_defaults = {
            self.name: {
                'simplified_no_remission_equations': False,
            },
        }
        
    @property
    def name(self):
        return self._name

    def setup(self, builder):
        """Load the disease prevalence and rates data."""
        self.data_prefix = 'chronic_disease.{}.'.format(self.name)
        self.prefix = self.name + '.'

        self.clock = builder.time.clock()
        self.start_year = builder.configuration.time.start.year
        self.simplified_equations = builder.configuration[self.name].simplified_no_remission_equations
        
        self.load_incidence(builder)
        self.load_remission(builder)
        self.load_mortality(builder)
        self.load_prevalence(builder)

        bau_scenario = 'BAU' 
        self.scenario = builder.configuration.scenario
        if self.scenario != bau_scenario:
            self.load_pifs(builder)
            self.register_incidence_modifier(builder)

        columns = []
        for rate in ['_S', '_C']:
            for when in ['', '_previous']:
                columns.append(self.name + rate + when)

        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=columns,
            requires_columns=['age', 'sex'])
        self.population_view = builder.population.get_view(columns)

        builder.event.register_listener(
            'time_step__prepare',
            self.on_time_step_prepare)

    def load_pifs(self,builder):
        model = builder.configuration.model

        pif_folder = 'pif_results/{}/{}/'.format(model, self.name)
        pif_filename = '{}_pifs_{}_{}.csv'.format(model, self.name, self.scenario)

        pif_data = pd.read_csv(pif_folder + pif_filename)
        pif_data.rename(columns = {'age': 'age_start', 'year':'year_start'}, inplace=True)
        pif_data['year_end'] = pif_data['year_start'] + 1
        pif_data['age_end'] = pif_data['age_start'] + 1
        
        self.pif = builder.lookup.build_table(pif_data, 
                                              key_columns=['sex'], 
                                              parameter_columns=['age','year'])

    
    def load_incidence(self, builder):
        inc_data = builder.data.load(self.data_prefix + 'incidence')
        i = builder.lookup.build_table(inc_data, 
                                       key_columns=['sex'], 
                                       parameter_columns=['age','year'])
        self.incidence = builder.value.register_rate_producer(
            self.prefix + 'incidence', source=i)


    def load_remission(self, builder):
        rem_data = builder.data.load(self.data_prefix + 'remission')
        r = builder.lookup.build_table(rem_data, 
                                       key_columns=['sex'], 
                                       parameter_columns=['age','year'])
        self.remission = builder.value.register_rate_producer(
            self.prefix + 'remission', source=r)


    def load_mortality(self, builder):
        mty_data = builder.data.load(self.data_prefix + 'mortality')
        f = builder.lookup.build_table(mty_data, 
                                       key_columns=['sex'], 
                                       parameter_columns=['age','year'])
        self.excess_mortality = builder.value.register_rate_producer(
            self.prefix + 'excess_mortality', source=f)


    def load_prevalence(self, builder):
        prev_data = builder.data.load(self.data_prefix + 'prevalence')
        self.initial_prevalence = builder.lookup.build_table(prev_data, 
                                                             key_columns=['sex'], 
                                                             parameter_columns=['age','year'])


    def register_incidence_modifier(self, builder):
        """Register that the disease incidence rate will be modified by 
        precaluclated pif values.

        Parameters
        ----------
        builder
            The builder object for the simulation, which provides
            access to event handlers and rate modifiers.

        """
        rate_name = self.prefix + 'incidence'
        modifier = lambda ix, inc_rate: self.incidence_adjustment(ix, inc_rate)
        builder.value.register_value_modifier(rate_name, modifier)

    def incidence_adjustment(self, index, incidence_rate):
        """Modify a disease incidence rate using pif values.

        Parameters
        ----------
        index
            The index into the population life table.
        incidence_rate
            The un-adjusted disease incidence rate.

        """
        new_rate = incidence_rate * (1 - self.pif(index))
        return new_rate


    def on_initialize_simulants(self, pop_data):
        """Initialize the test population for which this disease is modeled."""
        C = 1000 * self.initial_prevalence(pop_data.index)
        S = 1000 - C

        pop = pd.DataFrame({f'{self.name}_S': S,
                            f'{self.name}_C': C,
                            f'{self.name}_S_previous': S,
                            f'{self.name}_C_previous': C
                            }, index=pop_data.index)

        self.population_view.update(pop)

    def on_time_step_prepare(self, event):
        """
        Update the disease status for both the BAU and intervention scenarios.
        """
        # Do not update the disease status in the first year, the initial data
        # describe the disease state at the end of the year.
        if self.clock().year == self.start_year:
            return
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        idx = pop.index
        S, C = pop[f'{self.name}_S'], pop[f'{self.name}_C']

        # Extract all of the required rates *once only*.
        i = self.incidence(idx)
        r = self.remission(idx)
        f = self.excess_mortality(idx)

        # NOTE: if the remission rate is always zero, which is the case for a
        # number of chronic diseases, we can make some simplifications.
        if np.all(r == 0):
            r = 0
            if self.simplified_equations:
                # NOTE: for the 'mslt_reduce_chd' experiment, this results in a
                # slightly lower HALY gain than that obtained when using the
                # full equations (below).
                new_S = S * np.exp(- i)
                new_C = C * np.exp(- f) + S - new_S
                pop_update = pd.DataFrame({
                    f'{self.name}_S': new_S,
                    f'{self.name}_C': new_C,
                    f'{self.name}_S_previous': S,
                    f'{self.name}_C_previous': C
                }, index=pop.index)
                self.population_view.update(pop_update)
                return

        # Calculate common factors.
        i2 = i**2
        r2 = r**2
        f2 = f**2
        f_r = f * r
        i_r = i * r
        i_f = i * f
        f_plus_r = f + r

        # Calculate convenience terms.
        l = i + f_plus_r
        q = np.sqrt(i2 + r2 + f2 + 2 * i_r + 2 * f_r - 2 * i_f)
        w = np.exp(-(l + q) / 2)
        v = np.exp(-(l - q) / 2)

        # Identify where the denominators are non-zero.
        nz = q != 0
        denom = 2 * q

        new_S = S.copy()
        new_C = C.copy()

        # Calculate new_S, new_C, new_S_int, new_C_int.
        num_S = (2 * (v - w) * (S * f_plus_r + C * r)
                     + S * (v * (q - l)
                                + w * (q + l)))
        new_S[nz] = num_S[nz] / denom[nz]
        
        num_C = - ((v - w) * (2 * (f_plus_r * (S + C)
                                               - l * S)
                                          - l * C)
                       - (v + w) * q * C)
        new_C[nz] = num_C[nz] / denom[nz]
        
        pop_update = pd.DataFrame({
            f'{self.name}_S': new_S,
            f'{self.name}_C': new_C,
            f'{self.name}_S_previous': S,
            f'{self.name}_C_previous': C,
        }, index=pop.index)
        self.population_view.update(pop_update)

