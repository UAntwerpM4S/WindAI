from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/EGU26/NoPowerTF/checkpoint/NoPowerWind/inference-last.ckpt"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')