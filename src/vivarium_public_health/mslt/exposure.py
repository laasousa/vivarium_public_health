"""
===============
Exposures
===============

This module is for simulating an exposure table for a single exposure.

"""

import pandas as pd
import numpy as np

class Exposure:

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name
            The name of the exposure (e.g., ``"tobacco"``).
        """
        self._name = name
        self.configuration_defaults = {
            name: { 
                'constant_prevalence': False,
                'tax': False,
                'delay': 20,
            },
        }

    @property
    def name(self):
        return self._name

    def setup(self, builder):
        """Configure the delayed risk component.

        This involves loading the required data tables, registering event
        handlers and rate modifiers, and setting up the population view.
        """
        self.clock = builder.time.clock()
        self.load_config(builder)
        self.load_prevalence(builder)
        self.load_incidence(builder)
        self.load_remission(builder)
        self.load_mortality(builder)
        self.load_tax_effects(builder)
        self.register_exposure_acmr(builder)
                                                       
        # Add a handler to create the exposure bin columns.
        req_columns = ['age', 'sex', 'population']
        new_columns = self.get_bin_names()
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=new_columns,
            requires_columns=req_columns)

        # Add a handler to move people from one bin to the next.
        builder.event.register_listener('time_step__prepare',
                                        self.on_time_step_prepare)

        # Define the columns that we need to access during the simulation.
        view_columns = req_columns + new_columns
        self.population_view = builder.population.get_view(view_columns)
        

    def load_config(self,builder):
        self.config = builder.configuration
        self.start_year = self.config.time.start.year

        # Determine whether smoking prevalence should change over time.
        # The alternative scenario is that there is no remission; all people
        # who begin smoking will continue to smoke.
        self.constant_prevalence = self.config['exposure'][self.name]['constant_prevalence']

        self.tax = self.config['exposure'][self.name]['tax']

        self.bin_years = int(self.config['exposure'][self.name]['delay'])

    def load_prevalence(self,builder):
        # Load the initial prevalence.
        prev_data = pivot_load(builder,f'risk_factor.{self.name}.prevalence')
        self.initial_prevalence = builder.lookup.build_table(prev_data,
                                                             key_columns=['sex'], 
                                                             parameter_columns=['age','year'])

    def load_incidence(self,builder):
        # Load the incidence rates for the BAU and intervention scenarios.
        inc_data = builder.lookup.build_table(
            pivot_load(builder,f'risk_factor.{self.name}.incidence'), 
                       key_columns=['sex'], 
                       parameter_columns=['age','year']
        )
        inc_name = '{}.incidence'.format(self.name)
        inc_int_name = '{}_intervention.incidence'.format(self.name)
        self.incidence = builder.value.register_rate_producer(inc_name, source=inc_data)
        self.int_incidence = builder.value.register_rate_producer(inc_int_name, source=inc_data)

    def load_remission(self,builder):
        # Load the remission rates for the BAU and intervention scenarios.
        rem_df = pivot_load(builder,f'risk_factor.{self.name}.remission')
        # In the constant-prevalence case, assume there is no remission.
        if self.constant_prevalence:
            rem_df['remission'] = 0.0
        rem_data = builder.lookup.build_table(rem_df, 
                                              key_columns=['sex'], 
                                              parameter_columns=['age','year'])
        rem_name = '{}.remission'.format(self.name)
        rem_int_name = '{}_intervention.remission'.format(self.name)
        self.remission = builder.value.register_rate_producer(rem_name, source=rem_data)
        self.int_remission = builder.value.register_rate_producer(rem_int_name, source=rem_data)

    def load_mortality(self,builder):
        # We apply separate mortality rates to the different exposure bins.
        # This requires having access to the life table mortality rate, and
        # also the relative risks associated with each bin.
        self.acm_rate = builder.value.get_value('mortality_rate')
        mort_rr_data = pivot_load(builder,f'risk_factor.{self.name}.mortality_relative_risk')
        self.mortality_rr = builder.lookup.build_table(mort_rr_data, 
                                                       key_columns=['sex'], 
                                                       parameter_columns=['age','year'])

    def load_tax_effects(self,builder):
        # Load the effects of a tax on the exposure.
        tax_inc = pivot_load(builder,f'risk_factor.{self.name}.tax_effect_incidence')
        tax_rem = pivot_load(builder,f'risk_factor.{self.name}.tax_effect_remission')
        self.tax_effect_inc = builder.lookup.build_table(tax_inc, 
                                                         key_columns=['sex'], 
                                                         parameter_columns=['age','year'])
        self.tax_effect_rem = builder.lookup.build_table(tax_rem, 
                                                         key_columns=['sex'], 
                                                         parameter_columns=['age','year'])

    def register_exposure_acmr(self,builder):
        mortality_data = pivot_load(builder,'cause.all_causes.mortality')
        self.exposure_acmr = builder.value.register_rate_producer(
            '{}_acmr'.format(self.name), source=builder.lookup.build_table(mortality_data, 
                                                              key_columns=['sex'], 
                                                              parameter_columns=['age','year']))

    def get_bin_names(self):
        """Return the bin names for both the BAU and the intervention scenario.

        These names take the following forms:

        ``"name.no"``
            The number of people who have never been exposed.
        ``"name.yes"``
            The number of people currently exposed.
        ``"name.N"``
            The number of people N years post-exposure.

        The final bin is the number of people :math:`\ge N` years
        post-exposure.

        The intervention bin names take the form ``"name_intervention.X"``.
        """
        if self.bin_years == 0:
            delay_bins = [str(0)]
        else:
            delay_bins = [str(s) for s in range(self.bin_years + 2)]
        bins = ['no', 'yes'] + delay_bins
        bau_bins = ['{}.{}'.format(self.name, bin) for bin in bins]
        int_bins = ['{}_intervention.{}'.format(self.name, bin) for bin in bins]
        all_bins = bau_bins + int_bins
        return all_bins

    def on_initialize_simulants(self, pop_data):
        """
        Define the initial distribution of the population across the bins, in
        both the BAU and the intervention scenario.
        """
        # Set all bins to zero, in order to create the required columns.
        pop = pd.DataFrame({}, index=pop_data.index)
        for column in self.get_bin_names():
            pop[column] = 0.0

        # Update the life table, so that we can then obtain a view that
        # includes the population counts.
        self.population_view.update(pop)
        pop = self.population_view.get(pop_data.index)

        # Calculate the absolute prevalence by multiplying the fractional
        # prevalence by the population size for each cohort.
        # NOTE: the number of current smokers is defined at the middle of each
        # year; i.e., it corresponds to the person_years.
        bau_acmr = self.exposure_acmr.source(pop_data.index)
        bau_probability_of_death = 1 - np.exp(- bau_acmr)
        pop.population *= 1 - 0.5 * bau_probability_of_death

        prev = self.initial_prevalence(pop_data.index).mul(pop['population'], axis=0)
        self.population_view.update(prev)

        # Rename the columns and apply the same initial prevalence for the
        # intervention.
        bau_prefix = '{}.'.format(self.name)
        int_prefix = '{}_intervention.'.format(self.name)
        rename_to = {c: c.replace(bau_prefix, int_prefix)
                     for c in prev.columns if c.startswith(bau_prefix)}
        int_prev = prev.rename(columns=rename_to)
        self.population_view.update(int_prev)

    def on_time_step_prepare(self, event):
        """Account for transitions between bins, and for mortality rates.

        These transitions include:
        - New exposures
        - Cessation of exposure
        - Increased duration of time since exposure

        """
        if self.clock().year == self.start_year:
            return

        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        idx = pop.index
        bau_acmr = self.acm_rate.source(idx)
        inc_rate = self.incidence(idx)
        rem_rate = self.remission(idx)
        int_inc_rate = self.int_incidence(idx)
        int_rem_rate = self.int_remission(idx)

        # Identify the relevant columns for the BAU and intervention.
        bin_cols = self.get_bin_names()
        bau_prefix = '{}.'.format(self.name)
        int_prefix = '{}_intervention.'.format(self.name)
        bau_cols = [c for c in bin_cols if c.startswith(bau_prefix)]
        int_cols = [c for c in bin_cols if c.startswith(int_prefix)]

        # Extract the RR of mortality associated with each exposure level.
        mort_rr = self.mortality_rr(idx)

        # Normalise the survival rate; never-smokers should have a mortality
        # rate that is lower than the ACMR, since current-smokers and
        # previous-smokers have higher RRs of mortality.
        weight_by_initial_prevalence = True
        if weight_by_initial_prevalence:
            # Load the initial exposure distribution, because it will be used
            # to adjust the ACMR.
            prev = self.initial_prevalence(pop.index)
            prev = prev.loc[:, bau_cols]
            # Multiply these fractions by the RR of mortality associated with
            # each exposure level.
            bau_wtd_rr = prev.mul(mort_rr.loc[:, bau_cols])
        else:
            # Calculate the fraction of the population in each exposure level.
            bau_popn = pop.loc[:, bau_cols].sum(axis=1)
            bau_prev = pop.loc[:, bau_cols].divide(bau_popn, axis=0)
            # Multiply these fractions by the RR of mortality associated with
            # each exposure level.
            bau_wtd_rr = bau_prev.mul(mort_rr.loc[:, bau_cols])

        # Sum these terms to obtain the net RR of mortality.
        bau_net_rr = bau_wtd_rr.sum(axis=1)
        # The mortality rate for never-smokers is the population ACMR divided
        # by this net RR of mortality.
        bau_acmr_no = bau_acmr.divide(bau_net_rr)

        # NOTE: adjust the RR *after* calculating the ACMR adjustments, but
        # *before* calculating the survival probability for each exposure
        # level.
        penultimate_cols = [s + str(self.bin_years)
                            for s in [bau_prefix, int_prefix]]
        mort_rr.loc[:, penultimate_cols] = 1.0

        # Calculate the mortality risk for non-smokers.
        bau_surv_no = 1 - np.exp(- bau_acmr_no)
        # Calculate the survival probability for each exposure level:
        #     (1 - mort_risk_non_smokers)^RR
        bau_surv_rate = mort_rr.loc[:, bau_cols].rpow(1 - bau_surv_no, axis=0)
        # Calculate the number of survivors for each exposure level (BAU).
        pop.loc[:, bau_cols] = pop.loc[:, bau_cols].mul(bau_surv_rate)

        # Calculate the number of survivors for each exposure level
        # (intervention).
        # NOTE: we apply the same survival rate to each exposure level for
        # the intervention scenario as we used for the BAU scenario.
        rename_to = {c: c.replace('.', '_intervention.')
                     for c in bau_surv_rate.columns}
        int_surv_rate = bau_surv_rate.rename(columns=rename_to)
        pop.loc[:, int_cols] = pop.loc[:, int_cols].mul(int_surv_rate)

        # Account for transitions between bins.
        # Note that the order of evaluation matters.
        suffixes = ['', '_intervention']
        # First, accumulate the final post-exposure bin.
        if self.bin_years > 0:
            for suffix in suffixes:
                accum_col = '{}{}.{}'.format(self.name, suffix, self.bin_years + 1)
                from_col = '{}{}.{}'.format(self.name, suffix, self.bin_years)
                pop[accum_col] += pop[from_col]
        # Then increase time since exposure for all other post-exposure bins.
        for n_years in reversed(range(self.bin_years)):
            for suffix in suffixes:
                source_col = '{}{}.{}'.format(self.name, suffix, n_years)
                dest_col = '{}{}.{}'.format(self.name, suffix, n_years + 1)
                pop[dest_col] = pop[source_col]

        # Account for incidence and remission.
        col_no = '{}.no'.format(self.name)
        col_int_no = '{}_intervention.no'.format(self.name)
        col_yes = '{}.yes'.format(self.name)
        col_int_yes = '{}_intervention.yes'.format(self.name)
        col_zero = '{}.0'.format(self.name)
        col_int_zero = '{}_intervention.0'.format(self.name)

        inc = inc_rate * pop[col_no]
        int_inc = int_inc_rate * pop[col_int_no]
        rem = rem_rate * pop[col_yes]
        int_rem = int_rem_rate * pop[col_int_yes]

        # Account for the effects of a tax.
        if self.tax:
            # The tax has a scaling effect (reduction) on incidence, and
            # causes additional remission.
            tax_inc = self.tax_effect_inc(idx)
            tax_rem = self.tax_effect_rem(idx)
            int_inc = int_inc * tax_inc
            int_rem = int_rem + (1 - tax_rem) * pop[col_int_yes]

        # Apply the incidence rate to the never-exposed population.
        pop[col_no] = pop[col_no] - inc
        pop[col_int_no] = pop[col_int_no] - int_inc
        # Incidence and remission affect who is currently exposed.
        pop[col_yes] = pop[col_yes] + inc - rem
        pop[col_int_yes] = pop[col_int_yes] + int_inc - int_rem
        # Those who have just remitted enter the first post-remission bin.
        pop[col_zero] = rem
        pop[col_int_zero] = int_rem

        self.population_view.update(pop)

def pivot_load(builder, entity_key):
    """Helper method for loading dataframe from artifact.

    Performs a long to wide conversion if dataframe has an index column
    named 'measure'.

    """
    data = builder.data.load(entity_key)

    if 'measure' in data.columns :
        data  = data.pivot_table(index = [i for i in data.columns if i not in ['measure','value']], columns = 'measure', \
        values = 'value').rename_axis(None,axis = 1).reset_index()
    
    return data