from anemoi.utils.checkpoints import load_metadata, replace_metadata

path = "/mnt/weatherloss/WindPower/training/WPDistr/VeryHighCapacityGTFinetune/checkpoint/81daa05665cb4f4daf1452e60657465d/inference-last.ckpt"
metadata, arrays = load_metadata(path, supporting_arrays=True)
metadata['dataset']['variables_metadata'] = {}
replace_metadata(path, metadata, arrays)
print('Done.')