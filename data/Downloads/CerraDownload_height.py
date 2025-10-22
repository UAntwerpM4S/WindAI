#Script to download the cerra data at height levels 
import cdsapi

dataset = "reanalysis-cerra-height-levels"
request = {
    "variable": [
        "wind_direction",
        "wind_speed"
    ],
    "height_level": [
        "50_m",
        "100_m",
        "150_m",
        "200_m"
    ],
    "data_type": ["reanalysis"],
    "product_type": ["analysis"],
    "year": ["2021", "2022", "2023",
        "2024", "2025"],
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
client.retrieve(dataset, request, "Cerra_height.grib")