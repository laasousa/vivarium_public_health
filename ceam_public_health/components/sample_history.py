import pandas as pd

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns


class SampleHistory:
    """Collect a detailed record of events that happen to a sampled sub-population 
    
    For use with visualization or analysis. The records are written to an HDF file.
    """

    def __init__(self):
        if 'sample_history' not in config:
            config['sample_history'] = {}

        self.sample_size = config['sample_history'].getint('sample_size', 10000)
        self.output_path = config['sample_history'].get('path', '/tmp/sample.hdf')
        self.sample_frames = {}
        self.sample_index = []

    def setup(self, builder):
        self.randomness = builder.randomness('sample_history')

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        if self.sample_size is None or self.sample_size > len(event.index):
            self.sample_size = len(event.index)
        draw = self.randomness.get_draw(event.index)
        priority_index = [i for d,i in sorted(zip(draw,event.index), key=lambda x:x[0])]
        self.sample_index = priority_index[:self.sample_size]

    @listens_for('time_step__cleanup')
    @uses_columns(None)
    def record(self, event):
        sample = event.population.loc[self.sample_index]

        self.sample_frames[event.time] = sample

    @listens_for('simulation_end')
    def dump(self, event):
        # NOTE: I'm suppressing two very noisy warnings about HDF writing that I don't think are relevant to us
        import warnings
        import tables
        from pandas.core.common import PerformanceWarning
        warnings.filterwarnings('ignore', category=PerformanceWarning)
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        pd.Panel(self.sample_frames).to_hdf(self.output_path,
                                            key='/{}'.format(config['run_configuration']['configuration_name']))
