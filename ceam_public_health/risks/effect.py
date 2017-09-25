import numpy as np

import ceam_inputs as inputs

from vivarium.framework.population import uses_columns


def continuous_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for standard continuous risks.

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'
    tmrel = 0.5 * (risk.tmred.min + risk.tmred.max)
    max_exposure = risk.max_rr

    # FIXME: Exposure, TMRL, and Scale values should be part of the values pipeline system.
    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        exposure = np.minimum(population_view.get(rr.index)[exposure_column].values, max_exposure)
        relative_risk = np.maximum(rr.values**((exposure - tmrel) / risk.scale), 1)
        return rates * relative_risk

    return inner


def categorical_exposure_effect(risk):
    """Factory that makes functions which can be used as the exposure_effect for binary categorical risks

    Parameters
    ----------
    risk : `vivarium.config_tree.ConfigTree`
        The gbd data mapping for the risk.
    """
    exposure_column = risk.name+'_exposure'

    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        exposure_ = population_view.get(rr.index)[exposure_column]
        return rates * (rr.lookup(exposure_.index, exposure_))
    return inner


class RiskEffect:
    """RiskEffect objects bundle all the effects that a given risk has on a cause.
    """

    configuration_defaults = {
        'risks': {
            'apply_mediation': True,
        },
    }

    def __init__(self, risk, cause):
        self.risk = risk
        self.cause = cause

        # FIXME: I'm not taking the time to rewrite the stroke model right now, so unpleasant hack here.
        # -J.C. 09/05/2017
        self.cause_name = cause.name
        if cause == inputs.causes.ischemic_stroke or cause == inputs.causes.hemorrhagic_stroke:
            self.cause_name = 'acute_' + self.cause_name

        self.exposure_effect = (continuous_exposure_effect(self.risk) if self.risk.distribution != 'categorical'
                                else categorical_exposure_effect(self.risk))

    def setup(self, builder):
        self._rr_data = inputs.get_relative_risks(self.risk, self.cause, builder.configuration)
        self._paf_data = inputs.get_pafs(self.risk, self.cause, builder.configuration)
        self._mediation_factor = inputs.get_mediation_factors(self.risk, self.cause, builder.configuration)


        self.relative_risk = builder.lookup(self._rr_data)
        self.population_attributable_fraction = builder.lookup(self._paf_data)

        if builder.configuration.risks.apply_mediation:
            self.mediation_factor = builder.lookup(self._mediation_factor)
        else:
            self.mediation_factor = None

        builder.modifies_value(self.incidence_rates, '{}.incidence_rate'.format(self.cause_name))
        builder.modifies_value(self.paf_mf_adjustment, '{}.paf'.format(self.cause_name))

        return [self.exposure_effect]

    def paf_mf_adjustment(self, index):
        if self.mediation_factor:
            return self.population_attributable_fraction(index) * (1 - self.mediation_factor(index))
        else:
            return self.population_attributable_fraction(index)

    def incidence_rates(self, index, rates):
        if self.mediation_factor:
            return self.exposure_effect(rates, self.relative_risk(index).pow(1 - self.mediation_factor(index), axis=0))
        else:
            return self.exposure_effect(rates, self.relative_risk(index))

    def __repr__(self):
        return "RiskEffect(cause= {})".format(self.cause)


def make_gbd_risk_effects(risk):
    return [RiskEffect(risk=risk, cause=cause) for cause in risk.affected_causes]
