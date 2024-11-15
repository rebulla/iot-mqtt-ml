from ast import *
import pandas as pd
import asyncio
from enum import Enum
import json
from typing import Any
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Header, Path, Query, Request
from helpers import convert_features, predict_map
from mqtt_client import MQTTClient
from app_constants import ConnectionParams



# API description.
app = FastAPI(
    title="IoT UFMS: ML API for Traffic Light Control",
    description="**Python API to connect to devices and run ML commands.**",
    version="1.0.0",
)

class CoordinatesType(Enum):
    """Corrdinates Type Enum"""
    IMBIRUSSU = 0
    AFONSO_PENA = 1


class DayType(Enum):
    """Weekday Type Enum"""
    monday = "monday"
    tuesday = "tuesday"
    wednesday = "wednesday"
    thursday = "thursday"
    friday = "friday"
    saturday = "saturday"
    sunday = "sunday"


class TrafficIntensity(Enum):
    """Traffic Intensity"""
    Low = "Low"
    Moderate = "Moderate"
    Intense = "Intense"
    Very_Intense = "Very Intense"

# ------------ CLASSES --------------
class ApiResponse(BaseModel):
    """ApiResponse Object"""
    status: str
    payload: Any

class Command(BaseModel):
    """Command Object"""
    name: str = Field(description="Command Name")
    value: Any = Field(description="Set Value")

class Device(BaseModel):
    """Device Object"""
    device_id: str = Field(description="Device Id.")
    command_ack: bool = Field(description="Acknoledge response")
    commands: list[Command] = Field(description="List of Commands")

class PredictFeatures(BaseModel):
    """PredictFeatures Model"""
    time: str = Field(description="time format: HH:MM:SS")
    day: DayType = Field(description="Day string: monday, tuesday, wednesday, etc...")
    coordinates: CoordinatesType = Field(description="afonso_pena ou imbirussu")


# -----------------------------------

# ------------ API's ----------------
"""
@app.post("/execute")
async def execute_command(params: Device) -> ApiResponse:
    # # MQTT Configuration
    response_event = push_commands_to_device(params)
        
    try:
        # Wait for the acknowledgment and retrieve the payload
        if params.command_ack:
            response_payload = await asyncio.wait_for(response_event.get(), timeout=10)
            response = {"status": "success", "payload": response_payload}
            return ApiResponse(**response)
        else:
            response = {"status": "success", "payload": []}
            return ApiResponse(**response)
        
    except asyncio.TimeoutError:
        # Handle the case where no response payload was received within a timeout
        return JSONResponse(content={"status": "ack-timeout", "message": "device might be offline"}, status_code=404)
"""    

@app.post("/execute/prediction/{gateway_id}")
async def execute_prediction(gateway_id:str, prediction_params: PredictFeatures) -> ApiResponse:
    #Get Prediction
    afonso_pena_predict = get_predictions([prediction_params])
    #afonso_pena_predict, imbirussu_predict = get_predictions([prediction_params])
    #prediction = predict_map(afonso_pena_predict + imbirussu_predict)[0]
    prediction = predict_map(afonso_pena_predict)[0]
    device_name = None
    if prediction_params.coordinates.value == CoordinatesType.AFONSO_PENA.value:
        device_name = CoordinatesType.AFONSO_PENA.value
    else:
        device_name = CoordinatesType.IMBIRUSSU.value

    # format device command
    device = {
        "device_id": gateway_id + "::" + gateway_id,
        "command_ack": False,
        "commands": [
            {
                "name": "state",
                "value": prediction
            }
        ]
    }

    # send command to device
    params = Device(**device)
    response_event = push_commands_to_device(params)
        
    try:
        # Wait for the acknowledgment and retrieve the payload
        if params.command_ack:
            response_payload = await asyncio.wait_for(response_event.get(), timeout=10)
            response = {"status": "success", "payload": response_payload}
            return ApiResponse(**response)
        else:
            response = {"status": "success", "payload": prediction}
            return ApiResponse(**response)
        
    except asyncio.TimeoutError:
        # Handle the case where no response payload was received within a timeout
        return JSONResponse(content={"status": "ack-timeout", "message": "device might be offline"}, status_code=404)



@app.post("/predict")
def predict(params: list[PredictFeatures]) -> ApiResponse:

    afonso_pena_predict = get_predictions(params)
    predictions = afonso_pena_predict 
    predictions = predict_map(predictions)
    
    # format device command
    device = {
        "device_id": gateway_id + "::time",
        "command_ack": False,
        "commands": [
            {
                "name": "time",
                "value": time
            },
            {
                "name": "day",
                "value": day.value
            },
            {
                "name": "afonso_pena_traffic_intensity",
                "value": predictions
            },
            {
                "name": "imbirussu_traffic_intensity",
                "value": predictions
            }
        ]
    }

    # send command to device
    params = Device(**device)
    response_event = push_commands_to_device(params)
        
    try:
        response = {"status": "success", "payload": device}
        return ApiResponse(**response)
        
    except asyncio.TimeoutError:
        # Handle the case where no response payload was received within a timeout
        return JSONResponse(content={"status": "ack-timeout", "message": "device might be offline"}, status_code=404)



    #response = {"status": "success", "payload": predictions}
    #return ApiResponse(**response)


@app.post("/set_clock/{gateway_id}")
async def set_time(gateway_id:str, 
                   time: str = Query(..., description="gateway time", example="08:00:00"), 
                   day: DayType = Query(..., description="Day", example=DayType.sunday),
                   afonso_pena_traffic_intensity: int = Query(..., description="Traffic Intensity for Afonso Pena", example=1),
                   imbirussu_traffic_intensity: int = Query(..., description="Traffic Intensity for Imbirussu", example=0)) -> ApiResponse:
    # format device command
    device = {
        "device_id": gateway_id + "::time",
        "command_ack": False,
        "commands": [
            {
                "name": "time",
                "value": time
            },
            {
                "name": "day",
                "value": day.value
            },
            {
                "name": "afonso_pena_traffic_intensity",
                "value": afonso_pena_traffic_intensity
            },
            {
                "name": "imbirussu_traffic_intensity",
                "value": imbirussu_traffic_intensity
            }
        ]
    }

    # send command to device
    params = Device(**device)
    response_event = push_commands_to_device(params)
        
    try:
        # Wait for the acknowledgment and retrieve the payload
        if params.command_ack:
            response_payload = await asyncio.wait_for(response_event.get(), timeout=10)
            response = {"status": "success", "payload": response_payload}
            return ApiResponse(**response)
        else:
            response = {"status": "success", "payload": device}
            return ApiResponse(**response)
        
    except asyncio.TimeoutError:
        # Handle the case where no response payload was received within a timeout
        return JSONResponse(content={"status": "ack-timeout", "message": "device might be offline"}, status_code=404)






# --------------------------------------------

# ------------ HELPER METHODS ----------------
def get_predictions(params):
    afonso_pena_model = joblib.load("./models/afonso_model.joblib")
    imbirussu_model = joblib.load("./models/imbirussu_model.joblib")
    
    afonso_pena_features = []
    imbirussu_features = []
    for feature in params:
        if feature.coordinates == CoordinatesType.AFONSO_PENA:
            afonso_pena_features.append(convert_features(feature))
        elif feature.coordinates == CoordinatesType.IMBIRUSSU:
            imbirussu_features.append(convert_features(feature))

    afonso_pena_predict = []
    imbirussu_predict = []
    if len(afonso_pena_features) > 0:
        afonso_pena_df = np.array(afonso_pena_features)
        # Make predictions using the loaded model
        afonso_pena_predict = afonso_pena_model.predict(afonso_pena_df).tolist()

    if len(imbirussu_features) > 0:
        imbirussu_df = np.array(imbirussu_features)
        # Make predictions using the loaded model
        imbirussu_predict = imbirussu_model.predict(imbirussu_df).tolist()
    
    return afonso_pena_predict


def push_commands_to_device(params):
    # # MQTT Configuration
    MQTT_BROKER_PORT = 8883  # WSS MQTT over TLS port

    mqtt_client = MQTTClient(ConnectionParams.SERVER_URL, MQTT_BROKER_PORT, ConnectionParams.USERNAME, ConnectionParams.PASSWORD)

    # Set up the response callback and event
    response_event = asyncio.Queue()  # Use an asyncio.Queue to store the payloads

    def handle_response(message):
        try:
            payload = json.loads(message.decode())
             # Process the response here
            print("Received response:", payload)
            # Put the payload into the queue
            response_event.put_nowait(payload)
        except:
            print("_____ ERROR LOADING MQTT RESPONSE _____")

    #if params.command_ack:
    mqtt_client.set_response_callback(handle_response)
    mqtt_client.subscribe_response_topic(ConnectionParams.MASTER_TELEMETRY_TOPIC)
    
    # Send the command to the MQTT broker
    mqtt_client.send_command(ConnectionParams.MASTER_CONTROL_TOPIC, params.model_dump())
    return response_event

