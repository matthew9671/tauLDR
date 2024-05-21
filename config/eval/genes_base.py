import ml_collections

def get_config():

    model_location = '/home/groups/swl1/yixiuz/torch_fid/experiments/genes/2024-05-19/17-26-17_genes_baseline/checkpoints/ckpt_0000004999.pt'
    model_config_location = '/home/groups/swl1/yixiuz/torch_fid/experiments/genes/2024-05-19/17-26-17_genes_baseline/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'genes'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        # [['data', 'path'], pianoroll_dataset_path + '/train.npy'],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteGenes'
    data.path = '/home/groups/swl1/yixiuz/torch_fid/downloads/full_genes.pt'
    data.S = 64+1
    data.use_absorbing = True # The actual state size is S-1
    data.batch_size = 16
    data.shuffle = True
    data.shape = [60]

    config.sampler = sampler = ml_collections.ConfigDict()
    # We're going to override this for now
    sampler.name = 'PCTauLeapingBarker'
    sampler.num_steps = 500
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'absorbing'
    sampler.num_corrector_steps = 2
    sampler.corrector_step_size_multiplier = 1.
    sampler.corrector_entry_time = 0.9
    sampler.reject_multiple_jumps = True

    return config