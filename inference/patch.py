from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/WPDistr/ExtremelyHighCapacityGTFinetune/inference-anemoi-by_time-epoch_009-step_005000.ckpt"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')