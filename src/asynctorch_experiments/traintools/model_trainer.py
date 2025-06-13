from asynctorch_experiments.buildtools.model_builder import ModelBuilder
from asynctorch_experiments.buildtools.parameters import BuildParams, ExperimentParams
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle
from asynctorch_experiments.n_mnist.data import N_MNIST 
from torch.optim import Adam 
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import snntorch.functional as SF
from asynctorch.simulator.async_simulator import AsyncSimulator
from asynctorch_experiments.evaluation.monitor import Monitor, TestMonitor, MembraneMonitor, SpikeMonitor
from dataclasses import dataclass, field, asdict
import copy
import torch


class ModelTrainer:
    
    @staticmethod
    def pseudo_run(
        build_params:BuildParams,
        experiment_params:ExperimentParams,
        monitors:Monitor=[],
        n_passes:int=1,
        use_trained_model=False,
        pseudo_mode='test',
        pseudo_string=''
    ):
        F = build_params.F_train
        device = build_params.device
        
        train_mode = pseudo_mode=='train'
        
        print(f'({device}) Performing a pseudo run (F {F})')
        
        train_data, test_data, collate_fn = N_MNIST.load_data(
            build_params=build_params,
            train=train_mode,
            test=not train_mode
        )
        
        dataloader = DataLoader(
            train_data if train_mode else test_data, 
            batch_size=build_params.batchsize, 
            shuffle=train_mode, 
            collate_fn=collate_fn
        )
        if use_trained_model:
            model_dict = build_params.get_model_file().load()
            optim_dict = build_params.get_optim_file().load()
        else:
            model_dict=None
            optim_dict = None
            
        async_simulator, n_neurons = ModelBuilder.build_model(
            build_params=build_params,
            model_dict=model_dict,
            F_size=F
        )
        
        for monitor in monitors:
            monitor(
                async_simulator, 
                build_params=build_params,
                experiment_params=experiment_params,
                buffer_size = 256,
                pseudo_string=pseudo_string
            )
        

        loss_fn = SF.ce_rate_loss()
        accuracy_function = SF.accuracy_rate
        
        accuracies = []
        losses = []
        batch_sizes = []
        if train_mode:
            optim = Adam(
            params=async_simulator.parameters(), 
            lr=build_params.learning_rate, 
            weight_decay=build_params.weight_decay
        )

        
        if train_mode and optim_dict is not None:
            optim.load_state_dict(state_dict=optim_dict)
            # Make sure it's on the right device
            for state_id, state in optim.state.items():
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        state[k] = v.to(device)
            
            
        loss_fn = SF.ce_rate_loss()
        accuracy_function = SF.accuracy_rate
        
        accuracies = []
        losses = []
        
        if train_mode:
            
            
            
            async_simulator.train()
            
            passes = 0
            
            for data, targets in tqdm(dataloader):
                if passes == n_passes:
                    break
                data = data.to(device)
                targets = targets.to(device)
                async_simulator.reset_state() # Reset the state of the simulator (including the state of the neurons)

                # Forward pass
                ys = []
                for t in range(data.shape[0]):
                    ts_data = data[t].view(data[t].shape[0], -1)
                    spk_out = async_simulator(ts_data, dt=build_params.timestep_size)[:, n_neurons-build_params.n_outputs:]
                    ys.append(spk_out) 
                    
                    # perform logging
                    async_simulator._log_monitors()
                    
                y = torch.stack(ys)
                loss = loss_fn(y, targets)
                losses.append(loss.item())
                accuracy = accuracy_function(y, targets)
                accuracies.append(accuracy)

                # Backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()
                passes += 1
            
            
        else:
            async_simulator.eval()
        
            passes = 0
            
            with torch.inference_mode():
                for data, targets in tqdm(dataloader):
                    if passes == n_passes:
                        break
                    data = data.to(device)
                    targets = targets.to(device)
                    async_simulator.reset_state() # Reset the state of the simulator (including the state of the neurons)

                    # Forward pass
                    ys = []
                    for t in range(data.shape[0]):
                        ts_data = data[t].view(data[t].shape[0], -1)
                        spk_out = async_simulator(ts_data, dt=build_params.timestep_size)[:, n_neurons-build_params.n_outputs:]
                        ys.append(spk_out) 
                        
                        # perform logging
                        async_simulator._log_monitors()
                        
                        
                    y = torch.stack(ys)
                    loss = loss_fn(y, targets)
                    losses.append(loss.item())
                    accuracy = accuracy_function(y, targets)
                    accuracies.append(accuracy)
                    
                    batch_sizes.append(data.shape[0])
                    passes += 1

            test_acc = sum([a*b for a, b in zip(accuracies, batch_sizes)]) / sum(batch_sizes)
            test_loss = sum([a*b for a, b in zip(losses, batch_sizes)]) / sum(batch_sizes)
            print(f'\tacc: {test_acc:.3f}, loss: {test_loss:.3f}')
        # plot
        async_simulator._plot_monitors()
        
            
        

    
    
    @staticmethod
    def train_unit(
        previous_epoch_build_params: BuildParams,
        build_params: BuildParams
    ):
        device = build_params.device
        print(f'({device}) Training (F_train {build_params.F_train}; epoch {build_params.epoch+1})')

        train_data, _, collate_fn = N_MNIST.load_data(
            build_params=build_params,
            train=True,
            test=False
        )
        
        train_dataloader = DataLoader(
            train_data, 
            batch_size=build_params.batchsize, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        model_dict, optim_dict = None, None
        
        if previous_epoch_build_params is not None:
            model_dict = previous_epoch_build_params.get_model_file().load()
            optim_dict = previous_epoch_build_params.get_optim_file().load()
            
        async_simulator, n_neurons = ModelBuilder.build_model(
            build_params=build_params,
            model_dict=model_dict
        )
        
        optim = Adam(
            params=async_simulator.parameters(), 
            lr=build_params.learning_rate, 
            weight_decay=build_params.weight_decay
        )

        
        if not optim_dict is None:
            optim.load_state_dict(state_dict=optim_dict)
            # Make sure it's on the right device
            for state_id, state in optim.state.items():
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        state[k] = v.to(device)

        loss_fn = SF.ce_rate_loss()
        accuracy_function = SF.accuracy_rate
        
        accuracies = []
        losses = []
        
        async_simulator.train()
        
        for data, targets in tqdm(train_dataloader):
            data = data.to(device)
            targets = targets.to(device)
            async_simulator.reset_state() # Reset the state of the simulator (including the state of the neurons)

            # Forward pass
            ys = []
            for t in range(data.shape[0]):
                ts_data = data[t].view(data[t].shape[0], -1)
                spk_out = async_simulator(ts_data, dt=build_params.timestep_size)[:, n_neurons-build_params.n_outputs:]
                ys.append(spk_out) 
            y = torch.stack(ys)
            loss = loss_fn(y, targets)
            losses.append(loss.item())
            accuracy = accuracy_function(y, targets)
            accuracies.append(accuracy)

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        build_params.get_model_file().save(async_simulator.state_dict())
        build_params.get_optim_file().save(optim.state_dict())
        
        return np.array(accuracies), np.array(losses)
    
    
    
    @staticmethod
    def test_unit(
        build_params:BuildParams,
        F_test:int
    ) -> tuple[float, float]:
        
        device = build_params.device
        print(f'({device}) Testing F_test {F_test} (F_train {build_params.F_train}; epoch {build_params.epoch+1})')
        
        _, test_data, collate_fn = N_MNIST.load_data(
            build_params=build_params,
            train=False,
            test=True
        )
        
        test_dataloader = DataLoader(
            test_data, 
            batch_size=build_params.batchsize, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        model_dict = build_params.get_model_file().load()
            
        async_simulator, n_neurons = ModelBuilder.build_model(
            build_params=build_params,
            model_dict=model_dict,
            F_size=F_test
        )
        

        loss_fn = SF.ce_rate_loss()
        accuracy_function = SF.accuracy_rate
        
        accuracies = []
        losses = []
        batch_sizes = []
        
        async_simulator.eval()
        with torch.inference_mode():
            for data, targets in tqdm(test_dataloader):
                data = data.to(device)
                targets = targets.to(device)
                async_simulator.reset_state() # Reset the state of the simulator (including the state of the neurons)

                # Forward pass
                ys = []
                for t in range(data.shape[0]):
                    ts_data = data[t].view(data[t].shape[0], -1)
                    spk_out = async_simulator(ts_data, dt=build_params.timestep_size)[:, n_neurons-build_params.n_outputs:]
                    ys.append(spk_out) 
                y = torch.stack(ys)
                loss = loss_fn(y, targets)
                losses.append(loss.item())
                accuracy = accuracy_function(y, targets)
                accuracies.append(accuracy)
                
                batch_sizes.append(data.shape[0])

        test_acc = sum([a*b for a, b in zip(accuracies, batch_sizes)]) / sum(batch_sizes)
        test_loss = sum([a*b for a, b in zip(losses, batch_sizes)]) / sum(batch_sizes)
            
        print(f'\tacc: {test_acc:.3f}')
        return test_acc, test_loss
    