

from asynctorch.nn.architecture.mixed_architecture import AsyncLayer, AsyncNetwork, MixedArchitecture
import torch
from torch import nn
from typing import List, Tuple, Union
from math import prod

class DiscreteAsyncLayer(AsyncLayer):
    def __init__(
        self, 
        module: nn.Module, 
        from_input_indices: List[int], 
        to_neuron_indices: List[int], 
        n_neurons: int, 
        reshape_input_to: Union[Tuple[int, ...], None] = None
    ):
        super().__init__(
            module=module,
            from_input_indices=from_input_indices,
            to_neuron_indices=to_neuron_indices,
            n_neurons=n_neurons,
            reshape_input_to=reshape_input_to
        )


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs = inputs[:, self.from_input_indices]
        outputs = torch.zeros(batch_size, self.n_neurons, dtype=torch.float32, device=inputs.device)
        output_counts = torch.zeros_like(outputs)
        if inputs.count_nonzero() == 0:
            return outputs, output_counts
        output_counts[:, self.to_neuron_indices] = inputs.count_nonzero(dim=1).float().view(batch_size, 1)
        if self.reshape_input_to is not None:
            inputs = inputs.view(batch_size, *self.reshape_input_to)
        outputs[:, self.to_neuron_indices] = self.module(inputs).view(batch_size, len(self.to_neuron_indices))
        return outputs, output_counts
    

class DiscreteAsyncNetwork(AsyncNetwork):
    """
    A network of AsyncLayers.

    Args:
        input_layer: The input layer of the network. This layer takes the input spikes.
        network_layers: The layers of the network.
        n_neurons: The number of neurons in the network.
    """
    def __init__(
        self, 
        input_layer: DiscreteAsyncLayer, 
        network_layers: List[DiscreteAsyncLayer], 
        n_neurons: int
    ):
        super().__init__(
            input_layer=input_layer,
            network_layers=network_layers,
            n_neurons=n_neurons
        )

    def forward(self, s: torch.Tensor, is_input: bool) -> torch.Tensor:
        if is_input:
            currents, n_currents = self.input_layer(s)
        else:
            currents = torch.zeros(s.shape[0], self.n_neurons, dtype=torch.float32, device=s.device)
            n_currents = torch.zeros_like(currents)
            for layer in self.layers:
                layer_currents, layer_n_currents = layer(s)
                currents = currents + layer_currents
                n_currents = n_currents + layer_n_currents
        return currents, n_currents
    
    def build_sequential(module_per_layer: List[nn.Module], shape_per_layer: List[Tuple[int, ...]]):
        if len(module_per_layer) != len(shape_per_layer) - 1:
            raise RuntimeError("The number of layers must be equal to the number of input shapes minus one. Make sure to start with the input layer shape.")
        layers = []
        n_neurons = sum([prod(shape) for shape in shape_per_layer]) - prod(shape_per_layer[0])
        n_neurons_in_previous_layers = 0
        for i, module in enumerate(module_per_layer):
            if i == 0: # input layer
                input_indices = list(range(0, prod(shape_per_layer[0])))
                output_indices = list(range(0, prod(shape_per_layer[1])))
            else: # hidden layers
                n_neurons_in_layer = prod(shape_per_layer[i])
                n_neurons_in_next_layer = prod(shape_per_layer[i + 1])
                input_indices = list(range(n_neurons_in_previous_layers, n_neurons_in_previous_layers + n_neurons_in_layer))
                output_indices = list(range(n_neurons_in_previous_layers + n_neurons_in_layer, n_neurons_in_previous_layers + n_neurons_in_layer + n_neurons_in_next_layer))
                n_neurons_in_previous_layers += n_neurons_in_layer
            layer = DiscreteAsyncLayer(module, input_indices, output_indices, n_neurons, shape_per_layer[i])
            layers.append(layer)
        return DiscreteAsyncNetwork(layers[0], layers[1:], n_neurons)
    

class DiscreteMixedArchitecture(MixedArchitecture):
    def __init__(
            self, 
            async_network: DiscreteAsyncNetwork, 
            device: torch.device, 
            *args, 
            **kwargs
        ):
        super().__init__(
            async_network=async_network,
            device=device,
            *args, **kwargs
        )

    def forward(self, s: torch.Tensor, is_input: bool, is_input_and_neurons_combined: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_input_and_neurons_combined:
            raise RuntimeError("is_input_and_neurons_combined is not supported by this architecture. Input must be prioritized.")
        return self.async_network(s, is_input)
    