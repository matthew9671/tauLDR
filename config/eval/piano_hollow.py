import ml_collections

def get_config():

    pianoroll_dataset_path = '/home/groups/swl1/yixiuz/torch_fid/downloads/pianoroll_dataset'
    model_location = '/home/groups/swl1/yixiuz/torch_fid/experiments/piano/2024-05-14/00-36-56_piano_absorbing_hollow/checkpoints/ckpt_0000999999.pt'
    model_config_location = '/home/groups/swl1/yixiuz/torch_fid/experiments/piano/2024-05-14/00-36-56_piano_absorbing_hollow/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'piano'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        [['data', 'path'], pianoroll_dataset_path + '/train.npy'],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location
    config.pianoroll_dataset_path = pianoroll_dataset_path

    config.device = 'cpu'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'LakhPianoroll'
    data.path = pianoroll_dataset_path + '/train.npy'
    data.S = 129 + 1
    data.batch_size = 64 #128
    data.shuffle = True
    data.shape = [256]


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'ConditionalPCTauLeapingAbsorbingInformed' # ConditionalTauLeaping or ConditionalPCTauLeaping
    sampler.balancing_function = "mpf"
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'absorbing'
    sampler.test_dataset = pianoroll_dataset_path + '/test.npy'
    sampler.condition_dim = 32
    sampler.num_corrector_steps = 2
    sampler.corrector_step_size_multiplier = 0.1
    sampler.corrector_entry_time = 0.9
    sampler.reject_multiple_jumps = True

    return config