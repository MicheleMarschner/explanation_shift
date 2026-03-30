from src.configs.global_config import PATHS, ExperimentPaths
from src.evaluation.plots import main


def run_eval_pipeline(exp_config, stage, overwrite=False):

    exp_id = f"experiment__n{int(exp_config.N_PAIRS)}__{exp_config.expl_name}__seed{exp_config.seed}"
    exp_dir = ExperimentPaths.from_exp_dir(PATHS.runs / exp_id)
    save_dir = ExperimentPaths.from_exp_dir(PATHS.results / exp_id)

    if not exp_dir.exists():
        raise ValueError(f"train before evaluating")
    if save_dir.exists and not overwrite:
        print(f"{save_dir} already exists. Set overwrite if ")
        return
    
    main()

    
'''
    if stage == "reference":
        pass
    if stage == "artifact":
        #create_heatmaps(exp_dir, save_dir, corruption="gaussian_noise", severity=5)
    if stage == "drift":
        #sev_vs_corr_metric(exp_dir, save_dir, exp_config.SEVERITIES, corruption, slice="inv")
        #run_trust_zones_analysis(exp_dir, save_dir)
        #make_stage02_plots(exp_dir, save_dir, exp_config.CORRUPTIONS, exp_config.SEVERITIES, expl_metric, slice_name)
        pass
    if stage == "quantus":
        pass
'''

    