import tonic
import tonic.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64
from PIL import Image, ImageDraw
from asynctorch_experiments.buildtools.parameters import BuildParams
from asynctorch_experiments.buildtools.better_paths import MyPath
from asynctorch_experiments.data.load_data import AsyncDataSet
import numpy as np

import matplotlib.animation as animation

class N_MNIST(AsyncDataSet):
    def load_data(
            build_params: BuildParams,
            train: bool=False,
            test: bool=False
        ):
        input_shape = build_params.input_shape
        timestep_size=build_params.timestep_size
        project_path=build_params.project_path
        
        if train or test:
            return N_MNIST.resample_data(
                input_shape=input_shape,
                timestep_size=timestep_size,
                project_path=project_path,
                train=train,
                test=test,
                verbose=build_params.verbose
            )
        else: return None, None, None

    def resample_data(
        input_shape, 
        timestep_size, 
        project_path: MyPath, 
        train: bool,
        test: bool,
        verbose: bool
    ):
        data_path = project_path.change_dir('n_mnist', 'data_small')
        data_path = os.path.join(project_path, 'n_mnist', 'data_small')
        data_size_orig = tonic.datasets.NMNIST.sensor_size
        data_size_downsampled = (*input_shape[1:], input_shape[0])
        if verbose:
            print(f'resampling N_MNIST (in:{data_size_downsampled}, dt:{timestep_size})')
        transform = transforms.Compose([
            transforms.Downsample(sensor_size=data_size_orig, target_size=data_size_downsampled[:-1]),
            transforms.ToFrame(sensor_size=data_size_downsampled, time_window=timestep_size),
        ])
        train_dataset = tonic.datasets.NMNIST(
            save_to=data_path, 
            transform=transform, 
            train=True, 
            first_saccade_only=True
        ) if train else None
        test_dataset = tonic.datasets.NMNIST(
            save_to=data_path, 
            transform=transform, 
            train=False, 
            first_saccade_only=True
        ) if test else None
        
        collate_fn = tonic.collation.PadTensors(batch_first=False)
        return train_dataset, test_dataset, collate_fn
    

    def create_gif(sample, filename, project_path):
        processed_frames=[]
        n_frames=sample.shape[0]
        for i in range(n_frames):
            frame=np.zeros((*sample.shape[2:], 3),dtype=np.uint8)
            frame[:,:,0] = sample[i,0]*255
            frame[:,:,2] = sample[i,1]*255
            processed_frames.append(Image.fromarray(frame))
        processed_frames[0].save(os.path.join(project_path, 'gifs', f'{filename}.gif'), save_all=True, append_images=processed_frames[1:])


    def display_data(data, project_path):
        rows, cols = data.shape
        axs, fig = plt.subplots(rows, cols)

        for i in range(rows):
            for j in range(cols):
                sample, target = data[i,j]
                N_MNIST.create_gif(sample=sample, filename=f'{i}_{j}_test', project_path=project_path)


    def test_animated(frames):
        fig, ax = plt.subplots()
        t = np.linspace(0, 3, 40)
        g = -9.81
        v0 = 12
        z = g * t**2 / 2 + v0 * t

        v02 = 5
        z2 = g * t**2 / 2 + v02 * t

        scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
        line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
        ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
        ax.legend()


        def update(frame):
            # for each frame, update the data stored on each artist.
            x = t[:frame]
            y = z[:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            # update the line plot:
            line2.set_xdata(t[:frame])
            line2.set_ydata(z2[:frame])
            return (scat, line2)


        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
        plt.show()


        

