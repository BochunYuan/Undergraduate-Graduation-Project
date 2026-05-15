# Configuration for Ablation Experiment (消融组)
# Strategy: HLEU_MI_only -> Only the HLEU-conditioned MI branch
# This ablation removes Rep and Edge, keeping only MI to evaluate
# the contribution of the full composite scoring.

class ConfigS3DIS:
    chosen_rate_AL = 0.02
    al_iter = 0
    max_iter = 5
    active_strategy = 'HLEU_MI_only'  # Ablation: HLEU-conditioned MI only
    label_level = 3

    gpu = '0,1,2'
    max_steps = 60000
    stat_freq = 40
    save_freq = 1000
    input_channel = 6
    num_classes = 13
    ignore_idx = -100
    train_batch_size_mink = 2
    val_batch_size_mink = 8
    learning_rate = 1e-1
    ema_keep_rate = 0.955
    pseudo_threshold = 0.75
    optimizer = 'CosineAnnealingLR'
    save_ts_together = False
    propagate_with_uncert = True

    gaui_adv_mc_T = 10
    gaui_adv_mc_dropout_p = 0.2
    gaui_adv_rep_n_clusters = 10
    gaui_adv_fit_max_points = 50000
    gaui_adv_fit_sample_seed = 0
    gaui_adv_edge_eps = 1e-6

    score_parallel_jobs = 8
    score_required_levels = (0, 1, 2)
    score_grid_levels = (0.1, 0.5, 1.0)

    data_path = '/root/autodl-tmp/data_prepare/s3dis'
    label_path = '/root/autodl-tmp/HPAL_Final/labels'
    init_labeled_data = '/root/autodl-tmp/data_prepare/init_labeled/init_labeled20240525.json'
    base_path = '/root/autodl-tmp/Training_results/Ablation_experiment'

    saving_path = base_path + '/learner'
    model_save_dir_student = base_path + '/mink_pth_s'
    model_save_dir_teacher = base_path + '/mink_pth_t'
    labeled_save_path = base_path + '/labeled_data'
    save_path_feat = base_path + '/feat'
    save_path_probs = base_path + '/probs'
