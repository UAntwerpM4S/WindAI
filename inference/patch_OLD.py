from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/WindAI/VanillaPowerGT/checkpoint/630c9ccef176477c85eb935ad26435f6/inference-anemoi-by_time-epoch_029-step_200000.ckpt.bak"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')