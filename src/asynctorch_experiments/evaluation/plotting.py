import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
from asynctorch_experiments.buildtools.parameters import BuildParams, ExperimentParams

import colorsys


class Plotter:
    @staticmethod
    def save_fig_train(
        experiment_params: ExperimentParams
    ):
        file_name = experiment_params.get_train_figure_file()
        if not file_name.folder_exists():
            file_name.create_folder()
        file_name.save(None)
    
    def save_fig_test(
        experiment_params: ExperimentParams
    ):
        file_name = experiment_params.get_test_figure_file()
        if not file_name.folder_exists():
            file_name.create_folder()
        file_name.save(None)
        
    @staticmethod
    def plot_all_training_vals(
        ax1, ax2,
        train_accs, 
        train_losses,
        epoch_bounds,
        values,
    ):
        n_epochs = len(epoch_bounds)-1
        epochs = list(range(n_epochs+1))
        ax1.set_xticks(ticks=epochs, labels=epochs)
        ax2.set_xticks(ticks=epochs, labels=epochs)
        n_vals = train_accs.shape[0]

        window_size=20
        alpha = 0.3
        max_loss = train_losses.max()
        min_loss = train_losses.min()
        for val_idx in range(n_vals):

            train_accs_bounded = train_accs[val_idx, : , :, -window_size:].mean(axis=2)
            train_losses_bounded = train_losses[val_idx, : , :, -window_size:].mean(axis=2)
            train_accs_bounded_expaned = np.zeros((train_accs_bounded.shape[0]+1,train_accs_bounded.shape[1]))
            train_accs_bounded_expaned[1:,:] = train_accs_bounded
            train_losses_bounded_expaned = np.zeros((train_losses_bounded.shape[0]+1,train_losses_bounded.shape[1]))
            train_losses_bounded_expaned[1:,:] = train_losses_bounded
            train_losses_bounded_expaned[0,:] = max_loss
            

            ax2.plot(epochs, train_accs_bounded_expaned, c=f'C{val_idx+2}', linestyle='-', alpha=alpha, marker='o', markersize=6)
            ax1.plot(epochs, train_losses_bounded_expaned, c=f'C{val_idx+2}', linestyle='--', alpha=alpha, marker='x', markersize=6)


            ax2.plot(epochs, train_accs_bounded_expaned.mean(axis=1), c=f'C{val_idx+2}', linestyle='-', alpha=alpha*3, marker='o', markersize=10)
            ax1.plot(epochs, train_losses_bounded_expaned.mean(axis=1), c=f'C{val_idx+2}', linestyle='--', alpha=alpha*3, marker='x', markersize=10)        
        legend_labels= ['acc', 'loss', *values]
        legend_lines= [
            Line2D([0], [0], color='grey', marker='', linestyle='-',  lw=1),
            Line2D([0], [0], color=f'lightgrey', marker='', linestyle='--',  lw=1)
        ] + [
            Line2D([0], [0], color=f'C{i+2}', marker='', markersize=6, lw=1)
            for i in range(n_vals)
        ]
        ax1.legend(legend_lines, legend_labels, ncol=3, loc='upper center', frameon=False)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
    
    @staticmethod
    def plot_single_train_subplot(
        ax1,
        ax2,
        train_accs:np.ndarray, 
        train_losses:np.ndarray,
        epoch_bounds
        
    ):
        n_epochs = len(epoch_bounds)-1
        
        ax1.set_xticks(ticks=epoch_bounds, labels=list(range(n_epochs+1)))
        window_size = 20
        kernel = np.ones((20))/window_size

        d0, d1, d2 = train_accs.shape
        
        transposed_accs = train_accs.transpose((1,0,2))
        transposed_losses = train_losses.transpose((1,0,2))
        
        reshaped_accs = transposed_accs.reshape((d1, d2*d0))
        reshaped_losses = transposed_losses.reshape((d1, d2*d0))
        
        mean_reshaped_accs = reshaped_accs.mean(axis=0)
        mean_reshaped_losses = reshaped_losses.mean(axis=0)
        
        smoothed_mean_accs = np.convolve(mean_reshaped_accs, kernel, mode='valid')
        smoothed_mean_losses = np.convolve(mean_reshaped_losses, kernel, mode='valid')
        
        
        alpha = 0.3
        for epoch in range(n_epochs):
            start = epoch_bounds[epoch]
            end = epoch_bounds[epoch+1]
            xs = list(range(start, end))

            ax2.plot(xs, train_accs[epoch].T, c='C1', alpha=alpha)
            ax1.plot(xs, train_losses[epoch].T, c='C0', alpha=alpha)
        
        mean_xs = list(range(window_size-1, n_epochs * train_accs.shape[-1]))
        ax2.plot(mean_xs, smoothed_mean_accs, c='C1', alpha=alpha*3)
        ax1.plot(mean_xs, smoothed_mean_losses, c='C0', alpha=alpha*3)
            
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='C0')
        ax2.set_ylabel('Accuracy', color='C1')
        
        
        
    @staticmethod
    def plot_test_scores(
        experiment_params:ExperimentParams,
        test_accs:np.ndarray,
        test_losses:np.ndarray       
    ):
        param_name = experiment_params.param_name
        param_vals = experiment_params.values
        Fs_train = experiment_params.Fs_train
        Fs_test = experiment_params.Fs_test
        
        n_Fs_train, n_values, n_Fs_test, n_epochs, n_trials = experiment_params.get_full_experiment_shape()
        
        subplot_width = 4
        subplot_height = 4
        
        n_cols = n_values
        n_rows = n_epochs

        fig_width = n_cols*subplot_width+2
        fig_height = n_rows*subplot_height+2
        
        fig, axs = plt.subplots( n_rows, n_cols, figsize=(fig_width,fig_height))
        
        
        
        for i in range(n_rows):
            for j in range(n_cols):
                if n_rows == 1 and n_cols == 1:
                    ax = axs
                elif n_rows == 1:
                    ax = axs[j]
                elif n_cols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i,j]
                    
                for k, F_train in enumerate(Fs_train):
                    ax.plot(test_accs[k, j, :, i, :], c=f'C{k}', alpha=0.4)
                    ax.plot(test_accs[k, j, :, i, :].mean(axis=1), c=f'C{k}', label=F_train, marker='o', markersize=5)
                
                ax.grid()
                ax.set_xticks(list(range(n_Fs_test-1, -1, -1)), np.flip(Fs_test))
                ax.legend(ncol=3, loc='upper center', frameon=False)
                ax.set_xlabel(f'F_test')
                ax.set_ylabel(f'Accuracy')
                ax.set_ylim((0, 1.2))
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.set_title(f'{param_name}: {param_vals[j]} (epoch {i+1})')

        # plot best scores  per F_train per param per F_test (find best epoch)
        if n_rows > 1:
            for i, j, k in np.ndindex((n_Fs_train, n_values, n_Fs_test)):
                selection = test_accs[i,j,k,:,:]
                best_epoch = np.argmax(selection.mean(axis=1))
                if n_cols > 1:          
                    chosen_axis = axs[best_epoch, j]
                else: 
                    chosen_axis = axs[best_epoch]
                chosen_axis.plot(k, test_accs[i, j, k, best_epoch, :].mean(), c=f'C{i}', marker='*', markersize=15, markeredgecolor='k')
        plt.tight_layout()
        
        Plotter.save_fig_test(experiment_params=experiment_params)
        
        plt.plot()
            
    @staticmethod
    def plot_train_scores(
        experiment_params:ExperimentParams,
        train_accs:np.ndarray,
        train_losses:np.ndarray
    ):
        param_name = experiment_params.param_name
        param_vals = experiment_params.values
        Fs_train = experiment_params.Fs_train
        
        n_Fs_train, n_values, n_Fs_test, n_epochs, n_trials = experiment_params.get_full_experiment_shape()
        
        subplot_width = 4
        subplot_height = 4
        
        n_cols = n_values
        n_rows = n_Fs_train
        
        extra_row = n_rows>1
        extra_col = n_cols>1
        
        n_rows_expanded = n_rows + int(extra_row)
        n_cols_expanded = n_cols + int(extra_col)
        
        fig_width = n_cols_expanded*subplot_width+2
        fig_height = n_rows_expanded*subplot_height+2

        fig, axs = plt.subplots( n_rows_expanded, n_cols_expanded, figsize=(fig_width,fig_height))
        
        epoch_size = train_accs.shape[-1]
        epoch_bounds = np.arange(0, epoch_size*(n_epochs+1), epoch_size)

        axs_left = []
        axs_right = []
        
        if n_rows == 1 and n_cols == 1:
            ax2 = axs.twinx()
            Plotter.plot_single_train_subplot(
                ax1=axs,
                ax2=ax2,
                train_accs=train_accs[0, 0],
                train_losses=train_losses[0, 0],
                epoch_bounds=epoch_bounds
            )
            
        elif n_cols == 1:
            for i, ax1 in enumerate(axs):
                ax2 = ax1.twinx()
                axs_left.append(ax1)
                axs_right.append(ax2)
                if i < n_rows:
                    ax1.set_title(f'F_train: {Fs_train[i]}, value: {param_vals[0]}')
                    Plotter.plot_single_train_subplot(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs=train_accs[i, 0],
                        train_losses=train_losses[i, 0],
                        epoch_bounds=epoch_bounds
                    )
                else:
                    ax1.set_title(f'all Fs_train, value: {param_vals[0]}')
                    Plotter.plot_all_training_vals(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs = train_accs[:, 0],
                        train_losses=train_losses[:, 0] ,
                        epoch_bounds=epoch_bounds,
                        values=Fs_train
                    )
        elif n_rows == 1:
            for j, ax1 in enumerate(axs):
                ax2 = ax1.twinx()
                axs_left.append(ax1)
                axs_right.append(ax2)
                if j < n_cols:
                    ax1.set_title(f'F_train: {Fs_train[0]}, value: {param_vals[j]}')
                    Plotter.plot_single_train_subplot(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs=train_accs[0, j],
                        train_losses=train_losses[0, j],
                        epoch_bounds=epoch_bounds
                    )
                else:
                    ax1.set_title(f'F_train: {Fs_train[0]}, all values')
                    Plotter.plot_all_training_vals(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs = train_accs[0, :],
                        train_losses=train_losses[0, :],
                        epoch_bounds=epoch_bounds,
                        values=param_vals
                    )
        else:
            for i, j in np.ndindex(axs.shape):
                ax1 = axs[i,j]
                ax2 = ax1.twinx()
                axs_left.append(ax1)
                axs_right.append(ax2)
                if i < n_rows and j < n_cols:
                    ax1.set_title(f'F_train: {Fs_train[i]}, value: {param_vals[j]}')
                    Plotter.plot_single_train_subplot(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs=train_accs[i, j],
                        train_losses=train_losses[i, j],
                        epoch_bounds=epoch_bounds
                    )
                elif i == n_rows and j == n_cols:
                    for k in range(n_rows):
                        Plotter.plot_all_training_vals(
                            ax1=ax1,
                            ax2=ax2,
                            train_accs = train_accs[k, :],
                            train_losses=train_losses[k, :] ,
                            epoch_bounds=epoch_bounds,
                            values=param_vals
                        )
                elif i == n_rows:
                    ax1.set_title(f'all Fs_train, value: {param_vals[0]}')
                    Plotter.plot_all_training_vals(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs = train_accs[:, j],
                        train_losses=train_losses[:, j] ,
                        epoch_bounds=epoch_bounds,
                        values=Fs_train
                    )
                elif j == n_cols:
                    ax1.set_title(f'F_train: {Fs_train[0]}, all values')
                    Plotter.plot_all_training_vals(
                        ax1=ax1,
                        ax2=ax2,
                        train_accs = train_accs[i, :],
                        train_losses=train_losses[i, :],
                        epoch_bounds=epoch_bounds,
                        values=param_vals
                    )
                    
        if n_rows > 1 or n_cols > 1:
            # ---- Step 1: Compute global y-limits for left axes ----
            left_ymins = [ax.get_ylim()[0] for ax in axs_left]
            left_ymaxs = [ax.get_ylim()[1] for ax in axs_left]
            shared_left_ylim = (min(left_ymins), 1.2*max(left_ymaxs))

            # ---- Step 2: Compute global y-limits for right axes ----
            right_ymins = [ax.get_ylim()[0] for ax in axs_right]
            right_ymaxs = [ax.get_ylim()[1] for ax in axs_right]
            shared_right_ylim = (min(right_ymins), 1.2*max(right_ymaxs))

            # ---- Step 3: Apply shared limits ----
            for ax in axs_left:
                ax.set_ylim(shared_left_ylim)

            for ax in axs_right:
                ax.set_ylim(shared_right_ylim)
            
            
            fig.subplots_adjust(wspace=.3, hspace=-.3)
            plt.tight_layout()
            
        Plotter.save_fig_train(experiment_params=experiment_params)
        
        plt.show()
        
    @staticmethod
    def plot_experiment(
        experiment_params:ExperimentParams,
        train_accs:np.ndarray,
        train_losses:np.ndarray,
        test_accs:np.ndarray,
        test_losses:np.ndarray,
    ):

        Plotter.plot_train_scores(
            experiment_params=experiment_params,
            train_accs=train_accs, 
            train_losses=train_losses
        )
        
        Plotter.plot_test_scores(
            experiment_params=experiment_params,
            test_accs=test_accs,
            test_losses=test_losses
        )
