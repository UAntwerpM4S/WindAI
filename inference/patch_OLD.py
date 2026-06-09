from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/EGU26/BigTransformer/checkpoint/606532fc724149bcb7eb378f22d29d61/inference-anemoi-by_epoch-epoch_015-step_100000.ckpt"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')