import ml_collections

def get_config():
    save_directory = '/home/groups/swl1/yixiuz/torch_fid/experiments/aemet'

    config = ml_collections.ConfigDict()
    config.experiment_name = 'aemet_baseline'
    config.save_location = save_directory

    config.device = 'cuda'
    config.distributed = False
    config.num_gpus = 1

    config.loss = loss = ml_collections.ConfigDict()
    loss.name = 'GenericAux'
    loss.eps_ratio = 1e-9
    loss.nll_weight = 0.001
    loss.min_time = 0.01
    loss.one_forward_pass = True

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = 'Standard'
    training.n_iters = 5000 # Original paper used 5M parameters, 20 epochs of 10 iterations
    training.clip_grad = True
    training.warmup = 100

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteAEMET'
    data.path = '/home/groups/swl1/yixiuz/torch_fid/downloads/aemet.csv'
    data.S = 64+1
    data.use_absorbing = True # The actual state size is S-1
    data.batch_size = 73
    data.shuffle = True
    data.shape = [365]

    config.model = model = ml_collections.ConfigDict()
    model.name = 'AbsorbingSequenceTransformerEMA' # 'AbsorbingHollowSequenceTransformerFlashEMA'

    model.num_layers = 12
    model.d_model = 128
    model.num_heads = 16
    model.dim_feedforward = 2048
    model.dropout = 0.1
    model.temb_dim = 128
    model.num_output_FFresiduals = 2
    model.num_layers_per_mixed = 2
    model.time_scale_factor = 1000
    model.use_one_hot_input = True

    model.ema_decay = 0.9999

    # model.rate_const = 0.03
    model.rate_eps = 1e-3 # Instead of rate_const we have rate_eps for the log

    # model.sigma_min = 1.0
    # model.sigma_max = 100.0


    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = 'Adam'
    optimizer.lr = 2e-4

    config.saving = saving = ml_collections.ConfigDict()

    saving.enable_preemption_recovery = False
    saving.preemption_start_day_YYYYhyphenMMhyphenDD = None

    saving.checkpoint_freq = 500
    saving.num_checkpoints_to_keep = 2
    saving.checkpoint_archive_freq = 20000000
    saving.log_low_freq = 1000
    saving.low_freq_loggers = []
    saving.prepare_to_resume_after_timeout = False

    return config