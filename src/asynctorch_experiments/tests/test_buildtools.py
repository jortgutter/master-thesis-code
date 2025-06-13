from asynctorch_experiments.buildtools.parameters import BuildParams 
from asynctorch_experiments.n_mnist.experiment_dicts import EXPERIMENTS_NEW as EXPERIMENTS
from asynctorch_experiments.buildtools.model_builder import ModelBuilder
from asynctorch_experiments.buildtools.better_paths import MyPath, MyFile
from asynctorch_experiments.buildtools.model_unit import ModelUnit
import os

def get_project_path():
    testfile_path = MyPath(os.path.abspath(__file__))
    project_path = testfile_path.change_dir('../..')
    return project_path

def test_project_path():
    expected_path = 'asynctorch_experiments/src/asynctorch_experiments'
    project_path = get_project_path()
    print(project_path)
    print(expected_path)
    assert project_path.endswith(expected_path)
    
def test_change_dir():
    myPath = MyPath(f'test/bla/this')
    new_path1 = myPath.change_dir('..', '/that')
    new_path2 = myPath.change_dir('../that')
    assert new_path1 == new_path2
    assert new_path2 == 'test/bla/that'

def test_parameters():
    project_path = get_project_path()
    build_params_list, experiment_params = BuildParams.load_experiment(EXPERIMENTS, experiment_name='test', project_path=project_path)
    units = ModelUnit.load_units(build_params_list, experiment_params)
    unit_sh = experiment_params.get_unit_shape()
    exp_sh = experiment_params.get_experiment_shape()
    assert unit_sh[0] == exp_sh[0]
    assert unit_sh[1] == exp_sh[1]
    assert unit_sh[2] == exp_sh[3]
    
# def test_paths(project_path):
#     params = BuildParams.load_params(EXPERIMENTS, project_path=project_path, exp_name='test')
#     test_unit = ModelUnit(params, purge_old=False)
#     test_unit.save_meta_file()
#     test_unit.meta_file.test()
    

    