import cdsapi

dataset = "reanalysis-cerra-pressure-levels"
request = {
    "variable": [
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "pressure_level": [
        "500", "600", "700",
        "750", "800", "850",
        "900", "950", "1000"
    ],
    "data_type": ["reanalysis"],
    "product_type": ["analysis"],
    "year": [
        "2021", "2022", "2023",
        "2024", "2025"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "grib"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
