import os
import sys
import torch
import tonic
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import tonic.transforms as transforms
from asynctorch.simulator.async_simulator import AsyncSimulator
from asynctorch.nn.neuron.lif_state import LIFState
from asynctorch.nn.neuron.neuron_state import NeuronState
from asynctorch.simulator.extensions.spike_dropout_extension import SpikeDropoutExtension
from asynctorch.simulator.extensions.stop_on_output_extension import StopOnOutputExtension
from asynctorch.simulator.spike_scheduler import RandomSpikeScheduler, MomentumSpikeScheduler
from asynctorch.simulator.spike_selector import SpikeSelector
import snntorch as snn

from asynctorch_experiments.buildtools.parameters import BuildParams
# LIF imports
from asynctorch.nn.architecture.mixed_architecture import MixedArchitecture, AsyncNetwork
from asynctorch.nn.architecture.base_architecture import BaseArchitecture
#Mubrain imports
from asynctorch_experiments.buildtools.microbrain.mubrain_state import MubrainIFFunction, MubrainIFState
from asynctorch_experiments.buildtools.microbrain.discrete_mixed_architecture import DiscreteMixedArchitecture, DiscreteAsyncNetwork, DiscreteAsyncLayer

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.ao.nn.quantized as q_nn
import snntorch.functional as SF
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from asynctorch.utils.surrogate import ATan
from math import prod
from typing import NamedTuple



class SimplifiedTorch(nn.Module):
    def __init__(
        self,
        architecture: nn.Module,
        device:str = 'cpu'
    ):
        super().__init__()
        self.architecture=architecture
        self.device=device

    def forward(self, x):
        spike_rec = []
        membrane_rec = []
        utils.reset(self.architecture)
        for step in range(x.shape[0]):
            x_step = x[step]
            # print(f'step shape: {x_step.shape}')
            spk_out, mem_out = self.architecture(x[step])

            
            spike_rec.append(spk_out)
            membrane_rec.append(mem_out)

        return torch.stack(spike_rec), torch.stack(membrane_rec)


class ModelBuilder:
    @staticmethod
    def build_model(
            build_params: BuildParams,
            model_dict=None,
            F_size=0
        ) -> tuple[AsyncSimulator, int]:
        verbose = build_params.verbose
        
        if verbose:
            print(f'\n=================\nBuilding model...\n=================\n')
        
        assert build_params.neuron_model in ["lif", "mubrain"]
        assert build_params.scheduler in ['random', 'momentum']
        
        # build required torch modules
        module_per_layer, shape_per_layer, neurons_per_layer, n_neurons = ModelBuilder.build_linear_modules(build_params)
        
        # build the async network module
        network_module = ModelBuilder.build_async_network_module(
            build_params=build_params,
            shape_per_layer=shape_per_layer,
            module_per_layer=module_per_layer
        )
        
        # build state module
        neuron_state_module = ModelBuilder.build_state_module(
            build_params=build_params,
            neurons_per_layer=neurons_per_layer
        )

        # build spike selector 
        spike_selector = ModelBuilder.build_spike_selector(
            build_params=build_params,
            state_module=neuron_state_module,
            network_module=network_module,
            F_size=F_size
        )

        # Simulator and extensions
        forward_step_extensions=ModelBuilder.build_extensions(
            build_params=build_params,
            n_neurons=n_neurons
        )
        
        # create the async_simulator
        async_simulator = AsyncSimulator(
            neuron_state=neuron_state_module,
            spike_selector=spike_selector, 
            forward_step_extensions=forward_step_extensions
        )
        if not model_dict is None:
            if verbose:
                print(f'Loading state_dict')
            async_simulator.load_state_dict(model_dict) 
            
        async_simulator = async_simulator.to(device=build_params.device)
            
        return async_simulator, n_neurons
    
    
    @staticmethod
    def build_linear_modules(build_params:BuildParams):
        verbose = build_params.verbose

        linear_module = nn.Linear
        
        module_per_layer = []
        shape_per_layer = [(prod(build_params.input_shape), )]
        n_layers = len(build_params.layer_shapes)
        
        for i in range(n_layers):
            # build a single layer
            layer_in = prod(build_params.input_shape) if i == 0 else prod(build_params.layer_shapes[i-1])
            layer_out = prod(build_params.layer_shapes[i])
            
            layer = linear_module(
                layer_in,
                layer_out, 
                bias=False
            )

            module_per_layer.append(
                layer
            )
            
            shape_per_layer.append((layer_out, ))
            
        neurons_per_layer = [prod(shape) for shape in shape_per_layer[1:]]
        n_neurons = sum(neurons_per_layer)
        if verbose:
            print('created linear torch modules:')
            print('neurons_per_layer: ', neurons_per_layer)
            print('Number of neurons:', n_neurons)
        return module_per_layer, shape_per_layer, neurons_per_layer, n_neurons
    
    @staticmethod
    def build_async_network_module(
            build_params: BuildParams, 
            shape_per_layer:list,
            module_per_layer: list
        ) -> BaseArchitecture:
        
        verbose=build_params.verbose
        
        if build_params.neuron_model == 'mubrain':
            # build a discrete, quantized, microbrain network
            if verbose:
                print(f'\n======\nbuilding discrete mubrain network (precision: {build_params.quantize_weight_bits},{build_params.quantize_membrane_bits})\n======\n')
            
            async_network = DiscreteAsyncNetwork.build_sequential(
                module_per_layer=module_per_layer, 
                shape_per_layer=shape_per_layer
            ).to(build_params.device)
            
            network_module = DiscreteMixedArchitecture(
                async_network=async_network, 
                device=build_params.device
            )
            
        elif build_params.neuron_model == 'lif':
            # build a regular, LIF network
            if verbose:
                print('building regular lif network')
            async_network = AsyncNetwork.build_sequential(
                module_per_layer=module_per_layer, 
                shape_per_layer=shape_per_layer
            ).to(build_params.device)
            
            network_module = MixedArchitecture(
                async_network=async_network, 
                device=build_params.device
            )
        else:
            raise Exception(f'Invalid neuron model ({build_params.neuron_model})')
        
        if verbose:
            print('Async network module completed:')
            print(async_network)
            print('Number of parameters:', sum(p.numel() for p in async_network.parameters() if p.requires_grad))
        
        
        return network_module
    
    @staticmethod
    def build_state_module(
        build_params: BuildParams,
        neurons_per_layer: list
    ):
        verbose=build_params.verbose
        
        # spike gradient
        spike_grad = ATan(alpha=build_params.surrogate_alpha)
        if verbose:
            print(f'Created ATan surrogate spike gradient (alpha={build_params.surrogate_alpha})')
        
        if build_params.neuron_model == "lif":
            if verbose:
                print("Building lif state")
            state_module = LIFState(
                neurons_per_layer,
                tau_m=build_params.tau_m,
                membrane_threshold=build_params.threshold,
                spike_grad=spike_grad,
                device=build_params.device,
                refrac_dropout=build_params.refractory_dropout,
                backprop_threshold=build_params.backprop_threshold
            )
            
        elif build_params.neuron_model == 'mubrain':
            if verbose:
                print("Building mubrain state")
            state_module = MubrainIFState(
                neurons_per_layer,
                build_params.tau_m,
                build_params.threshold,
                spike_grad,
                build_params.device,
                refrac_dropout=build_params.refractory_dropout,
                backprop_threshold=build_params.backprop_threshold
            )
        else:
            raise Exception(f'Invalid neuron model ({build_params.neuron_model})')
        
        return state_module
        
    @staticmethod
    def build_spike_selector(
        build_params: BuildParams,
        state_module: NeuronState,
        network_module:BaseArchitecture,
        F_size: int = 0
    ) -> SpikeSelector:
        is_test = F_size > 0
        
        if not is_test:
            F_size = build_params.F_train
             
        verbose=build_params.verbose
        
        if verbose:
            print(f'Building spike selector ({build_params.scheduler} scheduler)')
            
        if build_params.scheduler == 'random':
            spike_scheduler = RandomSpikeScheduler(state_module)
        elif build_params.scheduler == 'momentum':
            spike_scheduler = MomentumSpikeScheduler(state_module, lambda_=build_params.momentum_noise)
        else:
            raise Exception(f'Invalid scheduler ({build_params.scheduler})')

        if verbose:
            print(f'F size = {F_size} (is_test = {is_test}, train F = {build_params.F_train})')
        
        spike_selector_module = SpikeSelector(
            architecture=network_module,
            spike_scheduler=spike_scheduler,
            forward_group_size=F_size, # How many spikes are processed in parallel per forward step
            device=build_params.device,
            prioritize_input=build_params.prioritize_input,
            log_queue_length=build_params.log_queue_length
        )
        
        return spike_selector_module
        
    @staticmethod
    def build_extensions(
        build_params: BuildParams,
        n_neurons: int
    ):
        forward_step_extensions = []
        
        # check for spike dropout extension
        do_input_spike_dropout = build_params.input_spike_dropout > 0.
        do_network_spike_dropout = build_params.network_spike_dropout > 0.
        if do_input_spike_dropout or do_network_spike_dropout:
            spike_dropout_p = max(build_params.input_spike_dropout, build_params.network_spike_dropout)
            forward_step_extensions.append(
                SpikeDropoutExtension(
                    p=spike_dropout_p, 
                    apply_to_input=do_input_spike_dropout,
                    apply_to_network=do_network_spike_dropout
                )
            )

        # check for StopOnOutputExtension: 
        if build_params.loss_fn == 'first_spike':
            forward_step_extensions.append(
                StopOnOutputExtension(
                    n_neurons=n_neurons,
                    n_outputs=build_params.n_outputs,
                    max_post_output_steps=build_params.max_post_output_steps
                )
            )
        return forward_step_extensions
        

    