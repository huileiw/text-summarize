import os
hps = dict()

hps['data_path'] = os.getcwd() + '/datafiles/'
hps['vocab_path'] = './datafiles/vocab'
hps['model_path'] = './models/'

# Important settings
hps['mode'] = 'train'
hps['single_pass'] = False

# Where to save output
hps['log_root'] = 'log/'
hps['exp_name'] = ''

# Hyperparameters
hps['hidden_dim'] = 128
hps['emb_dim'] = 128
hps['batch_size'] = 16
hps['max_enc_steps'] = 400
hps['max_dec_steps'] = 100
hps['beam_size'] = 4
hps['min_dec_steps'] = 35
hps['vocab_size'] = 50000
hps['lr'] = 0.15
hps['adagrad_init_acc'] = 0.1
hps['rand_unif_init_mag'] = 0.02
hps['trunc_norm_init_std'] = 1e-4
hps['max_grad_norm'] = 2.0

# Pointer-generator or baseline model
hps['pointer_gen'] = False

# Coverage hyperparameters
hps['coverage'] = False
hps['cov_loss_wt'] = 1.0

# Utility flags] = for restoring and changing checkpoints
hps['convert_to_coverage_model'] = False
hps['restore_best_model'] = False

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
hps['debug'] = False
