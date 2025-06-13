from asynctorch_experiments.buildtools.parameters import BuildParams, ExperimentParams
from asynctorch_experiments.traintools.model_trainer import ModelTrainer
from asynctorch_experiments.evaluation.plotting import  Plotter
from asynctorch_experiments.buildtools.model_unit import ModelUnit
from asynctorch_experiments.buildtools.better_paths import MyPath, MyFile

import copy

import numpy as np


class Experiment:
    
    @staticmethod
    def run_experiment(
        experiments:dict,
        exp_name:str,
        project_path: MyPath,
        purge_old=False,
        override_verbose=False,
        override_device=False,
        monitors=[],
        train_before_pseudo=False,
        pseudorun=False,
        pseudo_mode='test',
        pseudo_string=''
    ):
        build_params, experiment_params = BuildParams.load_experiment(
            experiments=experiments,
            experiment_name=exp_name,
            project_path=project_path,
            override_verbose=override_verbose,
            override_device=override_device
        )
        
        # shape = (Fs_train, param_vals, Fs_test, epochs, trials)
        full_exp_shape = experiment_params.get_full_experiment_shape()
        
        # shape = (Fs_train, param_vals, epochs, trials)
        train_score_shape =experiment_params.get_train_scores_shape()
        
        # shape = (Fs_train, param_vals, Fs_test, epochs, trials)
        test_score_shape = experiment_params.get_test_scores_shape()
        
        train_accs = None
        train_losses = None
        
        test_accs = np.zeros(test_score_shape)
        test_losses = np.zeros(test_score_shape)
        
        # prepare model units
        model_units = np.vectorize(
            lambda build_params: ModelUnit(
                build_params=build_params,
                purge_old=purge_old,
                monitors=monitors
            )
        )(build_params)
        
        n_units = np.prod(model_units.shape)
        
        if pseudorun:
            for unit_id, unit_idx in enumerate(np.ndindex(model_units.shape)):
                model_unit:ModelUnit = model_units[unit_idx]
                model_unit.pseudo_run(
                    experiment_params=experiment_params,
                    n_passes = pseudorun,
                    train_before_pseudo=train_before_pseudo,
                    pseudo_mode=pseudo_mode,
                    pseudo_string=pseudo_string
                )
        else:
        
            for unit_id, (f_tr, par, trial) in enumerate(np.ndindex(model_units.shape)): 
                print(
                    f'-> [unit {unit_id + 1}/{n_units}] - ' +
                    f'(F_train: {f_tr+1}/{full_exp_shape[0]}, '+
                    f'param: {par+1}/{full_exp_shape[1]}, ' + 
                    f'trial: {trial+1}/{full_exp_shape[4]})')
                for epoch in range(experiment_params.n_epochs):
                    for f_te, F_test in enumerate(experiment_params.Fs_test):
                        test_acc = model_units[f_tr, par, trial].get_test_acc(F_test, epoch)
                        test_loss = model_units[f_tr, par, trial].get_test_loss(F_test, epoch)
                        print(f'(ep {epoch+1}) {str(F_test).rjust(6)}: {test_acc:.3f} ({f_te+1}/{test_score_shape[2]})')
                        test_accs[f_tr, par, f_te, epoch, trial] = test_acc
                        test_losses[f_tr, par, f_te, epoch, trial] = test_loss
                    
                    print(f'-')
                    tr_acc = model_units[f_tr, par, trial].get_train_acc(epoch)
                    tr_loss = model_units[f_tr, par, trial].get_train_loss(epoch)
                    if train_accs is None or train_losses is None:
                        # initialize score arrays
                        train_accs = np.zeros((*train_score_shape, len(tr_acc)), dtype=np.ndarray)
                        train_losses = np.zeros((*train_score_shape, len(tr_loss)), dtype=np.ndarray)
                    train_accs[f_tr, par, epoch, trial,:] = tr_acc 
                    train_losses[f_tr, par, epoch, trial,:] = tr_loss
            
            experiment_params.save_parameter_json()
            
            Plotter.plot_experiment(
                experiment_params=experiment_params,
                train_accs=train_accs,
                train_losses=train_losses,
                test_accs=test_accs,
                test_losses=test_losses
            )
                
