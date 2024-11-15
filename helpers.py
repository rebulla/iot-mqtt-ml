


weekday_map = {'sunday': 0, 'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4, 'friday': 5, 'saturday': 6}
typical_traffic_map = {'Low': 0, 'Moderate': 1, 'Intense': 2, 'Very_Intense':3} 
position = {'imbirussu': 0, 'afonso': 1}

# Function to convert time stamp string to seconds
def time_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def convert_features(feature):
    position = feature.coordinates.value
    time = time_to_seconds(feature.time)
    day = weekday_map[str(feature.day.value)]

    return [position, time, day]

def predict_map(predictions):
    results = []
    for predict in predictions:
        if predict == 0:
            results.append('low')
        elif predict == 1:
            results.append('moderate')
        elif predict == 2:
            results.append('intense')
        else:
            results.append('very_intense')
    return results



# ml prediction template
# [
#   {
#     "time": "14:12:00",
#     "day": "sunday",
#     "afonso_pena_traffic_intensity": "Low",
#     "imbirussu_traffic_intensity": "Moderate"
#   }
# ]


