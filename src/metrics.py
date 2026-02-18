import quantus
import numpy as np

from typing import Dict, Any

def build_quantus_metrics() -> Dict[str, Any]:
    # Adapted from the Quantus climate tutorial defaults. :contentReference[oaicite:5]{index=5}
    metrics = {
        "complexity__sparseness": quantus.Sparseness(
            abs=True,
            normalise=False,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=True,
        ),
        "robustness__avg_sensitivity": quantus.AvgSensitivity(
            nr_samples=2,
            lower_bound=0.2,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
            abs=True,
            normalise=False,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=True,
        ),
        "faithfulness__corr": quantus.FaithfulnessCorrelation(
            nr_runs=10,
            subset_size=224,  # for CIFAR (32*32=1024), 224 is okay-ish; tune later
            perturb_baseline="black",
            perturb_func=quantus.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=True,
            normalise=False,
            return_aggregate=True,
            aggregate_func=np.mean,
            disable_warnings=True,
        ),
    }
    return metrics
