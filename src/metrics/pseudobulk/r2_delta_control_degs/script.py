import sys
from pathlib import Path

## VIASH START
par = {"prepared": "prepared.npz", "output": "output.h5ad"}
meta = {"name": "r2_delta_control_degs", "resources_dir": str(Path(__file__).resolve().parents[1] / "_shared")}
## VIASH END

sys.path.append(meta["resources_dir"])
from condition_centroid_metric_runner import run_condition_centroid_metric  # noqa: E402

run_condition_centroid_metric(meta["name"], par["prepared"], par["output"])
