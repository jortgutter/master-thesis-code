from asynctorch.simulator.async_simulator import AsyncSimulator
from asynctorch.nn.architecture.mixed_architecture import MixedArchitecture
from asynctorch.nn.architecture.mixed_architecture import AsyncLayer, AsyncNetwork
from asynctorch.nn.architecture.base_architecture import BaseArchitecture 
from asynctorch_experiments.buildtools.parameters import BuildParams, ExperimentParams
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import types

def get_neuron_shapes(async_simulator:AsyncSimulator):
    network = async_simulator.spike_selector.architecture.async_network
    input_layer = network.input_layer
    layers = network.layers
    n_neurons = network.n_neurons
    n_inputs = network.n_inputs
    neuron_counts = (input_layer.n_neurons-10, 10)

    assert neuron_counts[0] == n_neurons-10
    assert sum(neuron_counts) == n_neurons
    
    return neuron_counts
    
    
class Monitor:
    def __init__(
        self, 
        subject, 
        build_params:BuildParams, 
        experiment_params:ExperimentParams,
        buffer_size=-1,
        pseudo_string=''
    ):
        if buffer_size == -1:
            buffer_size=512
        self.buffer_size=buffer_size
        self.subject = subject
        self.build_params=build_params
        self.experiment_params=experiment_params
        self.registered=False
        self.name='abstract_monitor'
        self.figure_path = experiment_params.get_figure_folder_path()
        self.pseudo_string=pseudo_string
        
    def __post_init__(self):

        if not self.registered:
            self.register_monitor()
            self.register_functionality()
        self.registered=True

    def get_plot_name(self):
        return f'{self.name}_{self.pseudo_string}'

    def get_plots(self):
        return  []

    def log_hook(subject):
        
        for monitor in subject._monitor_names:
            subject._monitors[monitor].log()

    def plot_hook(subject):
        print('plotting monitors...')
        subplot_size = 4
        for monitor_name in subject._monitor_names:
            monitor: Monitor = subject._monitors[monitor_name]
            plot_func_list = monitor.get_plots()
            if len(plot_func_list) >0:
                subplot_shape = plot_func_list.shape
                if len(subplot_shape) > 1:
                    fig,axs = plt.subplots(*subplot_shape, figsize = [subplot_shape[0]*subplot_size, subplot_shape[1]*subplot_size])
                    for i in range(subplot_shape[0]):
                        for j in range(subplot_shape[1]):
                            ax = axs[i,j]
                            plot_func_list[i,j](ax)
                plt.tight_layout()
                param_name = monitor.experiment_params.param_name
                param_val = monitor.build_params.get_value(param_name)
                plt.suptitle(f'{param_name}: {param_val}')
                build_param_hash = monitor.build_params.hash()
                plot_name = monitor.get_plot_name()
                plot_file = monitor.figure_path.file(f'{plot_name}_{param_name}={param_val}_{build_param_hash}', extension='png')
                plot_file.save(None, force_create_path=True)
                plt.show()

    def getter_hook(subject):
        return subject._monitor_names, subject._monitors

    def register_monitor(self):
        monitor_name = self.name
        print(f'Registring {monitor_name}')
        subject=self.subject
        if not hasattr(subject, '_monitor_names'):
            subject._monitor_names = []
            subject._monitors = {}
        if self.name not in subject._monitor_names:
            subject._monitor_names.append(self.name)
            subject._monitors[self.name]=self
            subject._log_monitors = lambda: Monitor.log_hook(subject)
            subject._plot_monitors = lambda: Monitor.plot_hook(subject)
            subject._get_monitors = lambda: Monitor.getter_hook(subject)
            

    def register_functionality(self):
        pass
        
    def is_init(self):
        pass
            
    def deregister(self):
        subject = self.subject
        if hasattr(subject, '_monitor_names'):
            subject.monitors.remove(self.name)
            subject.monitors[self.name] = None
            

    def log(self):
        pass
    
    def plot(self):
        pass
    
class TestMonitor(Monitor):
    def __init__(self, subject, *args, **kwargs):
        super().__init__(subject, *args, **kwargs)
        self.subject.called_hook = 0
        self.subject.logged_numbers = []
        self.name='test monitor',

        super().__post_init__()
        
        
    def log(self):
        self.subject.called_hook += 1
        self.subject.logged_numbers.append(self.subject.my_value)
        
    def plot(self):
        subject = self.subject
        print(f'called {subject.called_hook} times: self.obj.logged_numbers')
        plt.plot(subject.logged_numbers)
        plt.show()
        
    def register_functionality(self):
        pass
        

class TestClass:
    def __init__(self, value, neuron_shapes=[3,2,6], batch_size = 4):
        
        self.my_value = value
        self.init_state=True
        self.neuron_state=self
        self.neuron_shapes = neuron_shapes
        self.n_neurons = sum(self.neuron_shapes)
        self.batch_size = batch_size
        self.spike_counts = torch.zeros((self.batch_size, self.n_neurons))
        self.membrane_potentials = torch.zeros((self.batch_size, self.n_neurons))
        self.pre_spike_membrane_potentials = torch.zeros((self.batch_size, self.n_neurons))
        self.threshold = 15
        
    def increment(self):
        self.init_state = True
        self.my_value += 1
        
    def add_potentials(self):
        self.membrane_potentials += np.random.randint(0, 4, size=(self.batch_size, self.n_neurons))

        self.pre_spike_membrane_potentials = self.membrane_potentials.clone()
        spikes = self.membrane_potentials >= self.threshold
        self.spike_counts += spikes
        self.membrane_potentials[spikes] = 0
        
    def is_init(self):
        return self.init_state
        
class MembraneMonitor(Monitor):
    pass

class WeightMonitor(Monitor):
    def __init__(
        self, 
        subject, 
        build_params:BuildParams, 
        experiment_params:ExperimentParams,
        buffer_size=-1,
        n_plot_rows = 5,
        pseudo_string=''
    ):
        super().__init__(
            subject,
            build_params,
            experiment_params,
            buffer_size=buffer_size,
            pseudo_string=pseudo_string
        )
        self.n_plot_rows = n_plot_rows
        self.neuron_counts = get_neuron_shapes(subject)
        self.buffer_idx = 0
        self.done_logging=False
        self.name = 'weight monitor'
        self.input_weight_buffer = None
        self.network_weight_buffers=None
        self.spike_buffer = None
        self.membrane_buffer=None
        self.membrane_prespike_buffer=None
        super().__post_init__()
        
    def log(self):
        if not self.is_init():
            self.attempt_init()
        subject = self.subject
        if self.is_init() and subject.is_init() and not self.done_logging:
            subject = self.subject
            # weights
            self.input_weight_buffer[self.buffer_idx] = self.input_layer.module.weight.data.detach().clone()
            for i, layer in enumerate(self.network_layers):
                self.network_weight_buffers[i][self.buffer_idx] = layer.module.weight.data.detach().clone()
            # spike and membrane buffers
            self.spike_buffer[self.buffer_idx] = subject.spike_counts.clone()

            neuron_state = subject.neuron_state
            self.membrane_buffer[self.buffer_idx] = neuron_state.membrane_potentials.clone()
            self.membrane_prespike_buffer[self.buffer_idx] = neuron_state.pre_spike_membrane_potentials.clone()
        

        # advance buffer idx
        self.buffer_idx = min(self.buffer_idx+1, self.buffer_size-1)
        if self.buffer_idx ==  self.buffer_size-1:
            self.done_logging=True
            
        

    def attempt_init(self):
        subject:AsyncSimulator=self.subject
        if subject.is_init():
            architecture: MixedArchitecture = subject.spike_selector.architecture
            network:AsyncNetwork = architecture.async_network
            self.input_layer = network.input_layer
            input_shape = self.input_layer.module.weight.data.shape
            self.network_layers: nn.ModuleList[AsyncLayer] = network.layers
            self.n_layers = len(self.network_layers)
            network_shapes = [layer.module.weight.data.shape for layer in self.network_layers]
            
            self.input_weight_buffer = torch.zeros((self.buffer_size, *input_shape))
            self.network_weight_buffers = [torch.zeros((self.buffer_size, *shape)) for shape in network_shapes]
            neuron_state = subject.neuron_state
            self.spike_buffer = torch.zeros((self.buffer_size, *subject.spike_counts.shape))
            self.membrane_buffer = torch.zeros((self.buffer_size, *neuron_state.membrane_potentials.shape))
            self.membrane_prespike_buffer = torch.zeros((self.buffer_size, *neuron_state.membrane_potentials.shape))
    
    
    def get_draw_function(self,row, col):
        start = self.starts[col]
        stop = self.stops[col]
        ts = self.ts
        
        plot_interval = ts[-1] // self.n_plot_rows
        capture_idx = row*plot_interval
        
        if col == 0:
            # inout weights
            data = self.input_weight_buffer
            name = self.names[0]

                
        else:
            data = self.network_weight_buffers[col-1]
            name = self.names[1][col-1]
            
        def draw(ax):
            ax.hist(data[capture_idx].flatten(), bins=128)
            ax.set_title(f'{name} (capture {capture_idx})')
            
        
        return draw
        
    
    
    def get_plots(self):
        self.ts = np.arange(0, self.buffer_idx) 
        
        neuron_cumsum = np.cumsum(self.neuron_counts)
        
        self.starts = [0, *neuron_cumsum[:-1]]
        self.n_cols = len(self.network_layers)+1
        self.n_rows = 5
        self.names = ['input weights', [f'layer {i} weights' for i in range(self.n_layers)]]
        #self.data = [self.membrane_prespike_buffer[0], self.membrane_buffer[0], self.spike_buffer[0]]
        self.stops=neuron_cumsum 
        draw_functions = [[self.get_draw_function(row, col) for col in range(self.n_cols)] for row in range(self.n_rows)]

        return np.array(draw_functions, dtype=object)
    
    def is_init(self):
        return self.input_weight_buffer is not None
        

class SpikeMonitor(Monitor):

    
    def __init__(
        self, 
        subject, 
        build_params:BuildParams, 
        experiment_params:ExperimentParams,
        buffer_size: int = -1,
        pseudo_string=''
    ):
        super().__init__(
            subject,
            build_params,
            experiment_params,
            buffer_size=buffer_size,
            pseudo_string=pseudo_string
        )
        self.neuron_counts = get_neuron_shapes(subject)
        self.buffer_idx = 0
        self.done_logging=False
        self.name = 'spike monitor'
        self.spike_buffer=None
        self.membrane_buffer=None
        self.membrane_prespike_buffer=None
        super().__post_init__()
        

    def register_functionality(self):
        subject=self.subject
        
        
    def attempt_init(self):
        subject=self.subject
        if subject.is_init():
            self.spike_buffer = torch.zeros((self.buffer_size, *subject.spike_counts.shape))
            neuron_state = subject.neuron_state
            self.membrane_buffer = torch.zeros((self.buffer_size, *neuron_state.membrane_potentials.shape))
            self.membrane_prespike_buffer = torch.zeros((self.buffer_size, *neuron_state.membrane_potentials.shape))
                
        
    def log(self):
        if not self.is_init():
            self.attempt_init()
        subject = self.subject
        if self.is_init() and subject.is_init() and not self.done_logging:
            subject = self.subject
            self.spike_buffer[self.buffer_idx] = subject.spike_counts.clone()

            neuron_state = subject.neuron_state
            self.membrane_buffer[self.buffer_idx] = neuron_state.membrane_potentials.clone()
            self.membrane_prespike_buffer[self.buffer_idx] = neuron_state.pre_spike_membrane_potentials.clone()
        # advance buffer idx
        self.buffer_idx = min(self.buffer_idx+1, self.buffer_size-1)
        if self.buffer_idx ==  self.buffer_size-1:
            self.done_logging=True
            
    def is_init(self):
        return self.spike_buffer is not None
        

    
    
    def get_draw_function(self,row, col):
        start = self.starts[col]
        stop = self.stops[col]
        ts = self.ts
        data = self.data[row][ts, start:stop]
        layer = col
        if row == 2:
            spikes = np.where(data>0)
            def draw(ax):
                ax.scatter(spikes[0], spikes[1], s=4)
                ax.set_title(f'{self.names[row]}\n(layer {layer})')
        else:
            def draw(ax):
                n_samples = data.shape[1]
                ax.plot(ts, data.detach().numpy(), alpha = 0.3)
                ax.set_title(f'{self.names[row]}\n(layer {layer})')
                
        return draw
    
    def get_plots(self):
        self.ts = np.arange(0, self.buffer_idx) 
        neuron_cumsum = np.cumsum(self.neuron_counts)
        self.starts = [0, *neuron_cumsum[:-1]]
        self.n_cols = len(self.neuron_counts)
        self.n_rows = 3
        self.data = [self.membrane_prespike_buffer[:,0,:], self.membrane_buffer[:,0,:], self.spike_buffer[:,0,:]]
        self.names = ['prespike membrane', 'postspike membrane', 'spike count']
        self.stops=neuron_cumsum 
        draw_functions = [[self.get_draw_function(row, col) for col in range(self.n_cols)] for row in range(self.n_rows)]

        return np.array(draw_functions, dtype=object)
    
    
    

    
    
    
class WeightDistributionMonitor(Monitor):
    pass


def main():
    test_obj = TestClass(5.)
    
    TestMonitor(test_obj)
    
    
    for i in range(100):
        test_obj.increment()
        test_obj.add_potentials()
        if i % 7 == 0 or i %5 == 0:
            test_obj._log_monitors()
    
    
if __name__ == '__main__':
  main()