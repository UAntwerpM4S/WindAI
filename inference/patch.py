from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/WPDistr/SemiHighCapacityGT/checkpoint/79c61ab7f58845249c7619a5d2ae3adc/inference-last.ckpt"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')