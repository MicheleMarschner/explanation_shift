import quantus
import numpy as np

from typing import Dict, Any


def build_quantus_metrics() -> Dict[str, Any]:
    # Adapted from the Quantus climate tutorial defaults. :contentReference[oaicite:5]{index=5}
    metrics = {
        "complexity__sparseness": quantus.Sparseness(
            abs=True,
            normalise=True,
            normalise_func=quantus.normalise_func.normalise_by_max,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=True,
            display_progressbar=True, 
        ),
        "robustness__avg_sensitivity": quantus.AvgSensitivity(
            nr_samples=50,  # maybe later 50, 20
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.functions.perturb_func.batch_uniform_noise,
            perturb_func_kwargs={"perturb_mean":0.0, "perturb_std":0.15},
            similarity_func=quantus.similarity_func.difference,
            abs=True, # im tut false
            return_nan_when_prediction_changes=True, # sollte ich das lassen
            normalise=True, # in einem false in einem true
            normalise_func=quantus.normalise_func.normalise_by_max,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True, 
        ),
        "faithfulness__corr": quantus.FaithfulnessCorrelation(
            nr_runs=100, # maybe later around 100, 10
            subset_size=300,  # for CIFAR (32*32=1024), 224 is okay-ish; tune later
            perturb_baseline="mean", # or black
            perturb_func=quantus.functions.perturb_func.batch_baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=True, # im tut false
            normalise=True,    # in einem true in einem false
            normalise_func=quantus.normalise_func.normalise_by_max,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True, 
        ),
        "robustness__local_lipschitz": quantus.LocalLipschitzEstimate(
            abs=True,
            normalise=True,
            normalise_func=quantus.normalise_func.normalise_by_max,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True,
        ),
        "randomisation__random_logit": quantus.RandomLogit(
            num_classes=10,
            seed=42,
            abs=True,
            normalise=True,
            normalise_func=quantus.normalise_func.normalise_by_max,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=False,
            display_progressbar=True,
        ),
    }
    return metrics





'''
Learn first how to use it - might be that model need to be randomly initialized etc. pp.
"randomisation__param_rnd": quantus.ModelParameterRandomisation(
            layer_order="independent",
            similarity_func=quantus.ssim,
            return_sample_correlation=True,
            abs=True,
            normalise=False,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=False,
        ),
"randomisation__rnd_logit": quantus.RandomLogit(
            num_classes=1000, # oder eher 10?
            similarity_func=quantus.ssim,
            abs=True,   # same
            normalise=True,    # same
            normalise_func=quantus.helpers.normalise_func.normalise_by_negative,
            aggregate_func=np.mean,
            return_aggregate=True,
            disable_warnings=True,
            display_progressbar=True, 
        ),
'''