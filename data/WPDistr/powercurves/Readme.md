# Power curves matching the turbine_types in the database csv file

## Infos
To automatically match the turbine_type in the database and the power curve file, ' ' and '/' in the turbine_types column have to be replaced by '_'

Files were created using https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/blob/master/py_wake/wind_turbines/generic_wind_turbines.py with the parameters (and default values for all other parameters):
- ws_cutin  = 4.0
- ws_cutout = 25.0
- turbulence_intensity = 0.05

