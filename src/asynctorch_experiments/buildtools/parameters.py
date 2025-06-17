
from dataclasses import dataclass, field, fields, asdict
import os
import copy
from collections.abc import Iterable
import hashlib
from abc import ABC, abstractmethod
from asynctorch_experiments.buildtools.better_paths import MyPath, MyFile
from asynctorch.simulator.async_simulator import AsyncSimulator
import numpy as np

def hash_string(input_string: str, hash_len: int=16) -> str:
    if hash_len <= 0 or hash_len > 256//4:
        raise ValueError(f"Hash length must be between 1 and {256//4} (not {hash_len})")

    # Hash and extract bytes
    full_digest = hashlib.sha256(input_string.encode()).hexdigest()
    
    return full_digest[:hash_len]

class VersionedObject(ABC):
    version: str
    def __init__(self, version: str):
        self.version = version

    def get_version(self):
        return self.version
    
    def hash(self, hash_len:int=16):
        if hash_len <= 0 or hash_len > 256//4:
            raise ValueError(f"Hash length must be between 1 and {256//4} (not {hash_len})")

        # Hash and extract bytes
        full_digest = hashlib.sha256(repr(self).encode()).hexdigest()
        
        return full_digest[:hash_len]

class StringList(list):
    def __str__(self):
        string_list = [str(i) for i in self]
        string=f"[{','.join(string_list)}]"
        return string
    
class StringTuple(tuple):
    def __str__(self):
        string_tuple = [str(i) for i in self]
        string=f"({','.join(string_tuple)})"
        return string
    
def hashtag(init=True): return field(metadata={"group": 'hash_field'}, init=init)

@dataclass
class Params(VersionedObject):
    # version field
    version: str = hashtag(init=False)
    
    def __post_init__(self):
        super().__init__(version='0.1.0')
    
    @classmethod
    def load_params(cls, config_dict):
        params = cls(**config_dict)
        return params
    
    def update_values(self, value_dict):
        verbose=value_dict['verbose'] if 'verbose' in value_dict else self.verbose
        for key, value in value_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if verbose:
                    print(f'Updated parameters: {key}->{value}')
            else:
                raise AttributeError(f"Invalid parameter: {key}")
    

    def hash(self):
        hash_fields = self.get_group(name='hash_field')
        return hash_string(str(hash_fields))
    
    def get_group(self, name):
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("group") == name
        }
        
    def __str__(self) -> str:
        str = '\n=================\nParameter Values\n=================\n\n'
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            str += (f"{name}:").ljust(20) + f" {getattr(self, field.name)}\n"
        return str

    def to_dict(self) -> dict:
        """Convert stored parameters to a dictionary."""
        return asdict(self)
    
    def get_value(self, name):
        return getattr(self, name)
    
    
@dataclass
class ExperimentParams(Params):
    experiment_name: str = hashtag()
    base_params: "BuildParams" = hashtag()
    param_name: str = hashtag()
    values: list = hashtag()
    Fs_train: list = hashtag()
    Fs_test: list = hashtag()
    n_trials: int = hashtag()
    n_epochs: int = hashtag()
    verbose: bool
    
    def get_train_scores_shape(self):
        n_Fs_train = len(self.Fs_train)
        n_values = len(self.values)
        n_trials = self.n_trials
        return (n_Fs_train, n_values, self.n_epochs, n_trials)
        
    def get_test_scores_shape(self):
        n_Fs_train = len(self.Fs_train)
        n_values = len(self.values)
        n_Fs_test = len(self.Fs_test)
        n_trials = self.n_trials
        return (n_Fs_train, n_values, n_Fs_test, self.n_epochs, n_trials)
    
    def get_unit_shape(self):
        n_Fs_train = len(self.Fs_train)
        n_values = len(self.values)
        n_trials = self.n_trials
        return (n_Fs_train, n_values, n_trials)
    
    def get_full_experiment_shape(self):
        n_Fs_train = len(self.Fs_train)
        n_values = len(self.values)
        n_Fs_test = len(self.Fs_test)
        n_trials = self.n_trials
        return (n_Fs_train, n_values, n_Fs_test, self.n_epochs, n_trials)
    
    def get_figure_folder_path(self):
        project_path = self.base_params.get_project_path()
        hash = self.hash()
        figure_folder = project_path.change_dir('evaluation', 'plots', f'{self.experiment_name}_{hash}')
        return figure_folder
    
    def get_train_figure_file(self):
        train_figure_file = self.get_figure_folder_path().file(
            file_name = 'training_curves',
            extension='png'
        )
        return train_figure_file
    
    def get_test_figure_file(self):
        test_figure_file = self.get_figure_folder_path().file(
            file_name = 'test_accuracies',
            extension='png'
        )
        return test_figure_file
    
    def save_parameter_json(self):
        param_dict = {
            'base_params': self.base_params.to_dict(),
            'overhead': self.to_dict()
        }
        param_dict['overhead'].pop('base_params')
        
        param_file = self.get_figure_folder_path().file(
            file_name="param_dict",
            extension="json"
        )
        if not param_file.folder_exists():
            param_file.create_folder()
        param_file.save(param_dict)
        

@dataclass
class BuildParams(Params):
    # network parameters
    simplified_torch: bool = hashtag()
    input_shape: StringList[int] = hashtag()
    layer_shapes: StringList[tuple[int]] = hashtag()
    n_outputs: int
    timestep_size: int = hashtag()
    scheduler: str = hashtag()
    loss_fn: str = hashtag()
    F_train: int = hashtag()
    Fs_test: list[int]
    max_post_output_steps: int = hashtag()
    # neuron parameters
    neuron_model: str = hashtag()
    threshold: float = hashtag()
    tau_m: int = hashtag()
    keep_modulo: bool = hashtag()
    quantize_weight_bits: int = hashtag()
    quantize_membrane_bits: int = hashtag()
    prioritize_input: bool = hashtag()
    log_queue_length: bool = hashtag()
    # data parameters
    dataset:str = hashtag()
    batchsize:int = hashtag()
    limit_max_spikes: int = hashtag()
    # training parameters
    backprop_threshold: float = hashtag()
    surrogate_alpha: float = hashtag()
    input_spike_dropout: float = hashtag()
    network_spike_dropout: float = hashtag()
    refractory_dropout: float = hashtag()
    momentum_noise: float = hashtag()
    learning_rate: float = hashtag()
    weight_decay: float = hashtag()
    epoch: int = hashtag()
    trial: int = hashtag()
    # overhead parameters
    verbose: bool
    device: str
    project_path: MyPath
    model_path: MyPath
    

    @staticmethod
    def load_experiment(
        experiments:dict,
        experiment_name:str,
        project_path:MyPath,
        override_verbose:bool=False,
        override_device:bool=False
    ) -> tuple[np.ndarray["BuildParams"], ExperimentParams]:
        print(f'Starting Experiment ({experiment_name})')
        
        # load default parameters
        base_param_dict = experiments['default']['build_params']
        overhead_dict = experiments['default']['overhead']
        base_param_dict['project_path'] = project_path
        
        # load experiment-specific parameters
        experiment_param_dict = experiments[experiment_name]['build_params']
        experiment_overhead_dict = experiments[experiment_name]['overhead']
        
        # combine
        base_param_dict.update(experiment_param_dict)
        overhead_dict.update(experiment_overhead_dict)
        
        # load overhead params
        overhead_dict['experiment_name'] = experiment_name
        overhead_dict['base_params'] = BuildParams.load_params(base_param_dict)
        experiment_params = ExperimentParams.load_params(overhead_dict)
        
        # possibly override verbosity to True
        if override_verbose:
            experiment_params.verbose=True

        # get the shape of the units array and initialize it
        unit_shape = experiment_params.get_unit_shape()
        unit_build_params = np.zeros(unit_shape, dtype=BuildParams)
        
        # create the units
        for i in range(unit_shape[0]):  # Fs_train
            for j in range(unit_shape[1]):  # param vals
                for k in range(unit_shape[2]):  # trials
                    unit_param_dict = copy.deepcopy(base_param_dict)
                    unit_param_dict['F_train'] = experiment_params.Fs_train[i]
                    unit_param_dict[experiment_params.param_name] = experiment_params.values[j]
                    unit_param_dict['Fs_test'] = experiment_params.Fs_test
                    unit_param_dict['epoch'] = experiment_params.n_epochs - 1
                    unit_param_dict['trial'] = k
                    if override_verbose:
                        # possibly override verbosity to True
                        unit_param_dict['verbose']=True
                        
                    # possibly override device
                    if override_device:
                        unit_param_dict['device'] = override_device

                    unit_build_params[i, j, k] = BuildParams.load_params(unit_param_dict)

        return unit_build_params, experiment_params
        
    
    def __post_init__(self):
        super().__post_init__()
        self.set_model_path()
        if self.verbose:
            print(self)


    def set_model_path(self):
        param_hash = self.hash()
        project_path = self.project_path
        if project_path is None or not project_path.exists:
            raise Exception('Pass valid project path when initializing parameters')
        self.model_path = project_path.change_dir(os.path.join('models', param_hash))
    
    def get_project_path(self) -> MyPath:
        return self.project_path
        
    def get_model_path(self) -> MyPath:
        param_hash = self.hash()
        project_path = self.project_path
        if project_path is None or not project_path.exists:
            raise Exception('Pass valid project path when initializing parameters')
        model_path = project_path.change_dir(os.path.join('models', param_hash))
        return model_path
    
    def get_meta_file(self) -> MyFile:
        file = self.get_model_path().file(
            file_name='meta_file',
            extension='pkl'
        )
        return file
    
    def get_model_file(self) -> MyFile:
        file = self.get_model_path().file(
            file_name='model',
            extension='pth'
        )
        return file
    
    def get_optim_file(self) -> MyFile:
        file = self.get_model_path().file(
            file_name='optim',
            extension='pth'
        )
        return file
        
    def has_been_trained(self) -> bool:
        return self.get_model_file().exists()
    
    def get_previous_epoch_params(self):
        if self.epoch == 0:
            return None
        previous_params = self.get_copy()
        previous_params.epoch -= 1
        return previous_params
        
    def set_device(self, device):
        self.device=device
        
    def get_copy(self) -> "BuildParams":
        return copy.deepcopy(self)

