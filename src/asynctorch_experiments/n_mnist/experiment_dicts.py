EXPERIMENTS = {
    'default': {
        'build_params':{
            'simplified_torch': False,
            'input_shape': (2,10,10),
            'layer_shapes': [(64, ), (10, )],
            'n_outputs': 10,
            'timestep_size': 20000,
            'scheduler': 'random',
            'loss_fn': 'rate',
            'F_train': '128',
            'Fs_test': ['128'],
            'max_post_output_steps': 0,
            # neuron parameters
            'neuron_model': 'lif',
            'threshold': 0.3,
            'tau_m': 1000,
            'keep_modulo': False,
            'quantize_weight_bits': 0,
            'quantize_membrane_bits': 0,
            'prioritize_input': True,
            'log_queue_length': False,
            # data parameters
            'dataset': 'N_MNIST',
            'batchsize': 64,
            # training parameters
            'backprop_threshold': None,
            'surrogate_alpha': 2.,
            'input_spike_dropout': 0.,
            'network_spike_dropout': 0.,
            'refractory_dropout': 0.8,
            'momentum_noise': 0.,
            'learning_rate': 1e-3,
            'weight_decay': 0.,
            'epoch': 0,
            'trial': 0,
            # overhead parameters
            'verbose':False,
            'device':'cpu',
            'project_path': None,
            'model_path': None
        },
        'overhead': {
            'param_name': 'timestep_size',
            'values': [10000,20000],
            'Fs_train': [128],
            'Fs_test':[128, 64, 32, 16, 8, 4, 2, 1],
            'n_trials': 1,
            'n_epochs': 1,
            'verbose': False
        }
    },
    'test': {
        'build_params':{
            'neuron_model': 'lif'
        },
        'overhead': {
            'param_name': 'timestep_size',
            'values': [50000, 100000],
            'Fs_train': [128, 8, 4],
            'Fs_test': [128, 64, 32, 16, 8, 4, 2, 1],
            'n_trials': 2,
            'n_epochs': 8,
            'verbose': False
        }
    },
    'mubrain_compare_clamp_low_to_0': {
        'build_params':{
            'timestep_size': 20000,
        },
        'overhead': {
            'param_name': 'neuron_model',
            'values': ['mubrain', 'lif'],
            'Fs_train': [128],
            'Fs_test': [128],
            'n_trials': 1,
            'n_epochs': 1,
            'verbose': False
        }
    },
    'mubrain_compare_quantized_membrane': {
        'build_params':{
            'timestep_size': 20000,
            'threshold': 1
        },
        'overhead': {
            'param_name': 'neuron_model',
            'values': ['mubrain', 'lif'],
            'Fs_train': [128],
            'Fs_test': [128],
            'n_trials': 1,
            'n_epochs': 1,
            'verbose': False
        }
    },
    'mubrain_longer_quantized_membrane_test': {
        'build_params':{
            'timestep_size': 20000,
            'threshold': 1
        },
        'overhead': {
            'param_name': 'neuron_model',
            'values': ['mubrain', 'lif'],
            'Fs_train': [128, 8, 4],

            'n_trials': 1,
            'n_epochs': 4,
            'verbose': False
        }
    },
    'mubrain_quantized_membrane_clamped_input': {
        'build_params':{
            'timestep_size': 20000,
            'threshold': 1
        },
        'overhead': {
            'param_name': 'neuron_model',
            'values': ['mubrain', 'lif'],
            'Fs_train': [128, 8, 4],

            'n_trials': 1,
            'n_epochs': 4,
            'verbose': False
        }
    }
}
