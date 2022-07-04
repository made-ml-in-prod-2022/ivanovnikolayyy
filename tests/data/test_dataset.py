from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.params import SplittingParams


def test_load_dataset(dataset_path: str):
    data, targets = read_data(dataset_path)
    assert len(data) > 10


def test_split_dataset(dataset_path: str):
    splitting_params = SplittingParams()
    data, targets = read_data(dataset_path)
    data_train, data_test, targets_train, targets_test = split_train_test_data(
        data, targets, splitting_params
    )
    assert data_train.shape[0] > 10
    assert data_test.shape[0] > 5
