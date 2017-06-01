from importlib import import_module

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal, norm

from ceam import config

from ceam.framework.population import uses_columns
from ceam.framework.event import listens_for
from ceam.framework.randomness import random

from ceam_inputs.gbd_mapping import risk_factors
import ceam_inputs as inputs


def assign_exposure_categories(df, susceptibility_column, categories):
    """Creates an 'exposure_category' column that assigns simulant's exposure based on their susceptibility draw.

    Parameters
    ----------
    df : pd.DataFrame
    susceptibility_column : pd.Series
    categories : list
        list of all of the category columns in df 
    """

    bool_list = [c + '_bool' for c in categories]
    for col in categories:
        df['{}_bool'.format(col)] = df['{}'.format(col)] < df[susceptibility_column]

    df['exposure_category'] = df[bool_list].sum(axis=1)

    # seems weird, but we need to add 1 to exposure category. e.g. if all values for a row
    # in bool_list are false that simulant will be in exposure cat1, not cat0
    df['exposure_category'] = df['exposure_category'] + 1
    df['exposure_category'] = 'cat' + df['exposure_category'].astype(str)

    return df[['exposure_category']]


def assign_relative_risk_value(df, categories):
    """Creates an 'relative_risk_value' column that assigns simulant's relative risk based on their exposure.

    Parameters
    ----------
    df : pd.DataFrame    
    categories : list
        list of all of the category columns in df
    """
    for col in categories:
        df.loc[(df['exposure_category'] == col, 'relative_risk_value')] = df[col]
    return df


def continuous_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    # FIXME: Exposure, TMRL, and Scale values should be part of the values pipeline system.
    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        return rates * np.maximum(
            rr.values**((population_view.get(rr.index)[exposure_column] - risk.tmrl) / risk.scale).values, 1)

    return inner


def categorical_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for binary categorical risks

    Parameters
    ----------
    risk : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        exposed = population_view.get(rr.index)[exposure_column]
        return rates * (rr.cat1.values**exposed)

    return inner


class RiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a cause.
    
    Parameters
    ----------
    rr_data : pandas.DataFrame
        A dataframe of relative risk data with age, sex, year, and rr columns
    paf_data : pandas.DataFrame
        A dataframe of population attributable fraction data with age, sex, year, and paf columns
    cause : `ceam.config_tree.ConfigTree`
        The gbd data mapping for the cause.
    exposure_effect : callable
        A function which takes a series of incidence rates and a series of
        relative risks and returns rates modified as appropriate for this risk
    """
    def __init__(self, rr_data, paf_data, cause, exposure_effect):
        self.rr_data = rr_data
        self.paf_data = paf_data
        self.cause = cause
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.rr_lookup = builder.lookup(self.rr_data)
        builder.modifies_value(self.incidence_rates, 'incidence_rate.{}'.format(self.cause.name))
        builder.modifies_value(builder.lookup(self.paf_data), 'paf.{}'.format(self.cause.name))

        return [self.exposure_effect]

    def incidence_rates(self, index, rates):
        return self.exposure_effect(rates, self.rr_lookup(index))

    def __repr__(self):
        return ("RiskEffect(rr_data= {},\npaf_data= {},\n".format(self.rr_data, self.paf_data)
                + "cause= {},\nexposure_effect= {})".format(self.cause, self.exposure_effect))


def uncorrelated_propensity(population):
    return random('initial_propensity', population.index)


def correlated_propensity(population, risk_factor):
    """Choose a propensity to the risk factor for each simulant that respects
    the risk factor's expected correlation with other risk factors unless there
    is no correlation data available in which case a uniformly distributed
    random propensity will be chosen.

    Parameters
    ----------
    population: pd.DataFrame
        The population to get propensities for. Must include 'sex' and 'age' columns.
    risk_factor: `ceam.config_tree.ConfigTree`
        The gbd data mapping for the risk.

    Notes
    -----
    This function does some calculation, including loading the correlation
    matrices, once for each risk where they could be done once for all
    risks. In practice this doesn't seem to cause meaningful performance
    problems and it simplifies the code significantly. It's something to be aware
    of though, especially for a model with high fertility or migration where
    this code may be run in each time step rather than primarily during
    initialization.
    """

    correlation_matrices = inputs.load_risk_correlation_matrices().set_index(
        ['risk_factor', 'sex', 'age']).sort_index(0).sort_index(1).reset_index()
    if risk_factor.name not in correlation_matrices.risk_factor.unique():
        # There's no correlation data for this risk, just pick a uniform random propensity
        return uncorrelated_propensity(population)

    risk_factor_idx = sorted(correlation_matrices.risk_factor.unique()).index(risk_factor.name)
    ages = sorted(correlation_matrices.age.unique())
    age_idx = (np.broadcast_to(population.age, (len(ages), len(population))).T
               >= np.broadcast_to(ages, (len(population), len(ages))))
    age_idx = np.minimum(len(ages) - 1, np.sum(age_idx, axis=1))

    qdist = norm(loc=0, scale=1)
    quantiles = pd.Series()

    seed = config.run_configuration.draw_number

    for (sex, age_idx), group in population.groupby((population.sex, age_idx)):
        matrix = correlation_matrices.query('age == @ages[@age_idx] and sex == @sex')
        del matrix['age']
        del matrix['sex']
        matrix = matrix.set_index(['risk_factor'])
        matrix = matrix.values

        dist = multivariate_normal(mean=np.zeros(len(matrix)), cov=matrix)
        draw = dist.rvs(group.index.max()+1, random_state=seed)
        draw = draw[group.index]
        quantiles = quantiles.append(
                pd.Series(qdist.cdf(draw).T[risk_factor_idx], index=group.index)
        )

    return pd.Series(quantiles, index=population.index)


def basic_exposure_function(propensity, distribution):
    """This function handles the simple common case for getting a simulant's
    based on their propensity for the risk. Some risks will require a more
    complex version of this.

    Parameters
    ----------
    propensity : pandas.Series
        The propensity for each simulant
    distribution : callable
        A function with maps propensities to values from the distribution
    """
    return distribution(propensity)


class ContinuousRiskComponent:
    """A model for a risk factor defined by a continuous value. For example
    high systolic blood pressure as a risk where the SBP is not dichotomized
    into hypotension and normal but is treated as the actual SBP measurement.
    
    Parameters
    ----------
    risk : ceam_inputs.gbd_mapping.risk_factors element
        The configuration data for the risk
    distribution_loader : callable
        A function which take a builder and returns a standard CEAM
        lookup table which returns distribution data.
    exposure_function : callable
        A function which takes the output of the lookup table created
        by distribution_loader and a propensity value for each simulant
        and returns the current exposure to this risk factor.
    """
    def __init__(self, risk, distribution_loader, exposure_function=basic_exposure_function):
        if isinstance(distribution_loader, str):
            module_path, _, name = distribution_loader.rpartition('.')
            distribution_loader = getattr(import_module(module_path), name)

        if isinstance(exposure_function, str):
            module_path, _, name = exposure_function.rpartition('.')
            exposure_function = getattr(import_module(module_path), name)

        self._risk = risk_factors[risk] if isinstance(risk, str) else risk
        self._distribution_loader = distribution_loader
        self.exposure_function = exposure_function

    def setup(self, builder):
        self.distribution = self._distribution_loader(builder)
        self.randomness = builder.randomness(self._risk.name)

        effect_function = continuous_exposure_effect(self._risk)

        risk_effects = make_gbd_risk_effects(self._risk, effect_function)

        self.population_view = builder.population_view([self._risk.name+'_exposure', self._risk.name+'_propensity'])

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['age', 'sex'])
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({
            self._risk.name+'_propensity': uncorrelated_propensity(event.population),
            self._risk.name+'_exposure': np.full(len(event.index), float(self._risk.tmrl)),
        }))

    @listens_for('time_step__prepare', priority=8)
    def update_exposure(self, event):
        population = self.population_view.get(event.index)
        distribution = self.distribution(event.index)
        new_exposure = self.exposure_function(population[self._risk.name+'_propensity'], distribution)
        self.population_view.update(pd.Series(new_exposure, name=self._risk.name+'_exposure', index=event.index))

    def __repr__(self):
        return "ContinuousRiskComponent(_risk= {}, distribution= {})".format(self._risk.name, self.distribution)


def make_gbd_risk_effects(risk, effect_function):
    # This is here to avoid a cyclic dependency with ceam_inputs.
    # Rather than significantly reorganizing the code to fix this now I'm
    # going to wait until we fully decouple this code from GBD access by
    # switching to bundled input data at which point this will just
    # vanish. -Alec 04/2017
    return [RiskEffect(
        inputs.get_relative_risks(risk_id=risk.gbd_risk, cause_id=cause.gbd_cause),
        inputs.get_pafs(risk_id=risk.gbd_risk, cause_id=cause.gbd_cause),
        cause,
        effect_function)
        for cause in risk.effected_causes]


class CategoricalRiskComponent:
    """A model for a risk factor defined by a dichotomous value. For example
    smoking as two categories: current smoker and non-smoker.
    Parameters
    ----------
    risk : ceam_inputs.gbd_mapping.risk_factors element
        The configuration data for the risk
    """
    def __init__(self, risk):
        self._risk = risk_factors[risk] if isinstance(risk, str) else risk

    def setup(self, builder):
        self.population_view = builder.population_view([self._risk.name+'_propensity', self._risk.name+'_exposure'])

        self.exposure = builder.value('{}.exposure'.format(self._risk.name))
        self.exposure.source = builder.lookup(inputs.get_exposures(risk_id=self._risk.gbd_risk))

        self.randomness = builder.randomness(self._risk.name)

        effect_function = categorical_exposure_effect(self._risk)
        risk_effects = make_gbd_risk_effects(self._risk, effect_function)

        return risk_effects

    @listens_for('initialize_simulants')
    @uses_columns(['age', 'sex'])
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({
            self._risk.name+'_propensity': uncorrelated_propensity(event.population),
            self._risk.name+'_exposure': np.full(len(event.index), False),
        }))

    @listens_for('time_step__prepare', priority=8)
    def update_exposure(self, event):
        pop = self.population_view.get(event.index)
        exposed = pop[self._risk.name+'_propensity'] < self.exposure(event.index).cat1
        self.population_view.update(pd.Series(exposed, name=self._risk.name+'_exposure'))

    def __repr__(self):
        return "CategoricalRiskComponent(_risk= {}, exposure= {})".format(self._risk.name, self.exposure.source)
