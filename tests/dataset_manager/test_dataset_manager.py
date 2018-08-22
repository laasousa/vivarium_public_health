from pathlib import Path
import random

import pytest
from vivarium.testing_utilities import build_table, metadata

from vivarium_public_health.dataset_manager.dataset_manager import (_subset_rows, _subset_columns, get_location_term,
                                                                    parse_artifact_path_config)


def test_subset_rows_extra_filters():
    data = build_table(1, 1990, 2010)
    with pytest.raises(ValueError):
        _subset_rows(data, missing_thing=12)


def test_subset_rows():
    values = [lambda *args, **kwargs: random.choice(['red', 'blue']),
              lambda *args, **kwargs: random.choice([1, 2, 3])]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'color', 'number'))

    filtered_data = _subset_rows(data, color='red', number=3)
    assert filtered_data.equals(data[(data.color == 'red') & (data.number == 3)])

    filtered_data = _subset_rows(data, color='red', number=[2, 3])
    assert filtered_data.equals(data[(data.color == 'red') & ((data.number == 2) | (data.number == 3))])


def test_subset_columns():
    values = [0, 'Kenya', 12, 35, 'red', 100]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'draw', 'location',
                                                    'age_group_start', 'age_group_end', 'color', 'value'))

    filtered_data = _subset_columns(data, keep_age_group_edges=False)
    assert filtered_data.equals(data[['age', 'year', 'sex', 'color', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=False, color='red')
    assert filtered_data.equals(data[['age', 'year', 'sex', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=True)
    assert filtered_data.equals(data[['age', 'year', 'sex', 'age_group_start', 'age_group_end', 'color', 'value']])

    filtered_data = _subset_columns(data, keep_age_group_edges=True, color='red')
    assert filtered_data.equals(data[['age', 'year', 'sex', 'age_group_start', 'age_group_end', 'value']])


def test_location_term():
    assert get_location_term("Cote d'Ivoire") == 'location == "Cote d\'Ivoire" | location == "Global"'
    assert get_location_term("Kenya") == "location == 'Kenya' | location == 'Global'"
    with pytest.raises(NotImplementedError):
        get_location_term("W'eird \"location\"")


def test_parse_artifact_path_config(base_config):
    artifact_path = Path(__file__).parent / 'artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}}, **metadata(str(Path('/'))))

    assert parse_artifact_path_config(base_config) == str(artifact_path)


def test_parse_artifact_path_relative_no_source(base_config):
    artifact_path = './artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}}, {})

    with pytest.raises(ValueError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_relative(base_config):
    base_config.update({'input_data': {'artifact_path': './artifact.hdf'}}, **metadata(__file__))

    assert parse_artifact_path_config(base_config) == str(Path(__file__).parent / 'artifact.hdf')


def test_parse_artifact_path_config_fail(base_config):
    artifact_path = Path(__file__).parent / 'not_an_artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}}, **metadata(str(Path('/'))))

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_config_fail_relative(base_config):
    base_config.update({'input_data': {'artifact_path': './not_an_artifact.hdf'}}, **metadata(__file__))

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)

