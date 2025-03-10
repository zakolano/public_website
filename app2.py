from called_functions.prediction_functions import *

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
from called_functions.modelfunctions import *
from called_functions.helper_functions import *
from called_functions.other_info import *
from fastapi import WebSocket
import asyncio
from called_functions.group_assignments import *
import redis.asyncio as redis  # Use the async Redis client
import json
import uuid  # For unique session IDs
import numpy as np
from pydantic import BaseModel
import pickle
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from called_functions.model_wrapper import main_distribution_with_progress
from fastapi import BackgroundTasks
from functools import partial

from fastapi import BackgroundTasks

class ParameterSet(BaseModel):
    parameters: List[str]
    year: int
    name: str

class SelectedPreset(BaseModel):
    name: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

model_df = pd.read_csv("all_data_2025.csv")
model_df2 = model_df.copy()
years = [year for year in range(2011, 2025) if year != 2020]

def helper(x):
    return "_".join(x.split("_")[:-1])
c = pd.Series(model_df.columns).apply(helper).drop_duplicates()

async def get_session_id(request: Request):
    """Retrieve or generate a unique session ID for each user."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())  # Generate a new unique session ID
    return session_id

async def get_session_data(session_id: str):
    """Retrieve session data for the specific session ID from Redis."""
    session_data = await redis_client.get(f"session_data:{session_id}")
    if session_data:
        session_data = json.loads(session_data)
        if 'model' in session_data:
            # Decode and deserialize the model
            if session_data['model'] is not None:
                encoded_model = session_data['model']
                serialized_model = base64.b64decode(encoded_model)
                session_data['model'] = pickle.loads(serialized_model)
        print('session_data found')
        return session_data
    print('session_data not found')
    return {
        'teams': [],
        'selected_parameters': [],
        'year': None,
        'model_type': None,
        'hyperparameters': {},
        'predictions': {},
        'variable_display_names': stats_dict,
        'sd_params': ["_".join(x.split('_')[:-1]) for x in model_df.columns.tolist()[:-1] if x.split('_')[-1] == 'SD'],
        'prediction_method': None,
        'round_correct': None,
        'score': 0,
        'loaded_params': False,
        'loaded_model': False,
        'saved_parameter_sets': [],
        'selected_preset': None,
        'model':None, 
        'group_stuff': None, 
        'score_stuff' : None, 
        'prob_stuff' : None, 
        'score' : None
    }

async def save_session_data(session_id: str, session_data):
    """Save session data to Redis for the specific session ID."""
    # Convert any ndarray objects to lists before saving
    def convert_ndarray_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: convert_ndarray_to_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_ndarray_to_list(item) for item in data]
        else:
            return data

    session_data = convert_ndarray_to_list(session_data)

    # Serialize the model if it exists in session_data
    if 'model' in session_data and session_data['model'] is not None:
        serialized_model = pickle.dumps(session_data['model'])
        session_data['model'] = base64.b64encode(serialized_model).decode('utf-8')

    await redis_client.set(f"session_data:{session_id}", json.dumps(session_data))

def group_columns_by_section(columns):
    sections = {}
    for col in columns:
        section = col.split("_")[-1]  # Extract last part of column name
        if section not in sections:
            sections[section] = []
        sections[section].append(col)
    return sections

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    #print('Session data retrieved:', session_data)
    data_parameters = group_columns_by_section(c[:-1])
    selected_year = session_data.get('year', years[-1])  # Set default year to 2024
    selected_year = 2024 if selected_year is None else selected_year
    models = ["Neural Network", "Decision Tree", "Linear Regression"]

    show_modal = not session_data.get('loaded_params', False)
    if show_modal:
        session_data['loaded_params'] = True
        await save_session_data(session_id, session_data)
    print(f"show_modal: {show_modal}")
    #show_modal = True
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "years": years,
        "data_parameters": data_parameters,
        "models": models,
        "selected_year": selected_year,
        "selected_parameters": session_data.get('selected_parameters', []),
        "sd_parameters": session_data['sd_params'],
        "variable_display_names": stats_dict,
        "section_display_names": section_dict, 
        "show_modal": show_modal, 
        "saved_parameter_sets": session_data.get('saved_parameter_sets', []),
        "selected_preset": session_data.get('selected_preset', None)
    })
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/", response_class=HTMLResponse)
async def post_year_selection(request: Request, year: int = Form(...)):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    session_data['year'] = year
    await save_session_data(session_id, session_data)
    return RedirectResponse(url="/", status_code=303)

@app.post("/save_parameters", response_class=HTMLResponse)
async def save_parameters(request: Request, features: List[str] = Form(...), year: int = Form(...), preset: str = Form(None)):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    session_data['selected_parameters'] = features
    session_data['year'] = year
    session_data['selected_preset'] = preset
    await save_session_data(session_id, session_data)
    return RedirectResponse(url="/model_selection", status_code=303)

@app.get("/model_selection", response_class=HTMLResponse)
async def get_model_selection(request: Request):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    models = ["Neural Network", "Decision Tree", 'XGBoost', 'Random Forest', "Linear Regression", 'Nearest Neighbors']
    show_modal2 = not session_data.get('loaded_model', False)
    if show_modal2:
        session_data['loaded_model'] = True
        await save_session_data(session_id, session_data)
    print(f"show_modal2: {show_modal2}")
    return templates.TemplateResponse("model_selection.html", {
        "parameters": session_data['selected_parameters'],
        "request": request,
        "models": models,
        "selected_model": session_data.get('model_type', ''),
        "selected_method": session_data.get('prediction_method', ''),
        "hyperparameters": session_data.get('hyperparameters', {}),
        "show_modal2": show_modal2
    })
import base64
@app.post("/process_model", response_class=HTMLResponse)
async def process_model(
    request: Request,
    model_type: str = Form(...),
    learning_rate: float = Form(None),
    epochs: int = Form(None),
    max_depth: int = Form(None),
    min_samples_split: int = Form(None),
    fit_intercept: str = Form(None), 
    n_neighbors: int = Form(None),
    k_weights: str = Form(None),
    n_estimators: int = Form(None),
    loss_function: str = Form(None),
    num_models: int = Form(None),
    criterion: str = Form(None),
    prediction_method: str = Form(None),
    subsample: float = Form(None)
):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    session_data['model_type'] = model_type
    session_data['prediction_method'] = prediction_method
    hyperparameters = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'fit_intercept': fit_intercept, 
        'n_neighbors': n_neighbors,
        'k_weights': k_weights,
        'n_estimators': n_estimators,
        'loss_function': loss_function,
        'num_models': num_models,
        'criterion': criterion, 
        'max_depth': max_depth, 
        'subsample': subsample
    }
    if not session_data.get('selected_parameters'):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No parameters selected. Please go back and select parameters.",
            "redirect_url": "/"
        })
    if not session_data.get('model_type'):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No model type selected. Please go back and select a model type.",
            "redirect_url": "/model_selection"
        })
    session_data['hyperparameters'] = {k: v for k, v in hyperparameters.items() if v is not None}
    
    # Save the session data and redirect to loading page
    await save_session_data(session_id, session_data)
    return templates.TemplateResponse("loading-template.html", {"request": request, "session_id": session_id})

@app.get("/results", response_class=HTMLResponse)
async def get_results(request: Request):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    
    # Check if results are already computed
    if 'predictions' in session_data and session_data['predictions'] is not None:
        # Results are ready, display them
        return templates.TemplateResponse("results.html", {
            "request": request,
            "prediction_method": session_data['prediction_method'],
            'groups': session_data['group_stuff'],
            'scores': session_data['score_stuff'],
            'probs': session_data['prob_stuff'],
            'bracket_score': session_data['score'],
            'round_correct': session_data['round_correct'],
            'apply_button': 1
        })
    else:
        # Redirect back to loading page if someone tries to access results directly
        return templates.TemplateResponse("loading-template.html", {"request": request, "session_id": session_id})
    
@app.post("/results", response_class=HTMLResponse)
async def post_model_selection(
    request: Request,
    model_type: str = Form(...),
    learning_rate: float = Form(None),
    epochs: int = Form(None),
    max_depth: int = Form(None),
    min_samples_split: int = Form(None),
    fit_intercept: str = Form(None), 
    n_neighbors: int = Form(None),
    k_weights: str = Form(None),
    n_estimators: int = Form(None),
    loss_function: str = Form(None),
    num_models: int = Form(None),
    criterion: str = Form(None),
    prediction_method: str = Form(None),
    subsample: float = Form(None)
):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    session_data['model_type'] = model_type
    session_data['prediction_method'] = prediction_method
    hyperparameters = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'fit_intercept': fit_intercept, 
        'n_neighbors': n_neighbors,
        'k_weights': k_weights,
        'n_estimators': n_estimators,
        'loss_function': loss_function,
        'num_models': num_models,
        'criterion': criterion, 
        'max_depth': max_depth,
        'subsample': subsample
    }
    if not session_data.get('selected_parameters'):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No parameters selected. Please go back and select parameters.",
            "redirect_url": "/"
        })
    if not session_data.get('model_type'):
        return templates.TemplateResponse("results.html", {
            "request": request,
            "error": "No model type selected. Please go back and select a model type.",
            "redirect_url": "/model_selection"
        })
    session_data['hyperparameters'] = {k: v for k, v in hyperparameters.items() if v is not None}

    # Here, you can integrate your ML model and pass these values for processing
    if session_data["teams"] is None:
        session_data["teams"] = ["Team1", "Team2"]
    year = session_data['year']
    print(model_type)
    session_data['hyperparameters']['epochs'] = 100
    session_data['hyperparameters']['learning_rate'] = 0.01
    predictions_dict, model2 = main_distribution(session_data['hyperparameters'],  session_data['model_type'], session_data['prediction_method'], session_data['year'], session_data['selected_parameters'])
    serialized_model = pickle.dumps(model2)
    model = base64.b64encode(serialized_model).decode('utf-8')
    groups_stuff, round_correct, score = assign_groups(predictions_dict['winners'], session_data['year'])

    session_data['model'] = model
    session_data['predictions'] = predictions_dict
    session_data['round_correct'] = round_correct
    await save_session_data(session_id, session_data)

    if(prediction_method == 'score'):
        score_stuff = assign_score(predictions_dict['Score'])
    else:
        score_stuff = None

    if('RandomForest' in str(model)):
        prob_stuff = assign_prob(predictions_dict['probabilities'])
    else:
        prob_stuff = None
    session_data['group_stuff'] = groups_stuff
    session_data['score_stuff'] = score_stuff
    session_data['prob_stuff'] = prob_stuff
    session_data['score'] = score
    await save_session_data(session_id, session_data)
    return templates.TemplateResponse("results.html", {"request": request,  "prediction_method": session_data['prediction_method'], 'groups': groups_stuff, 'scores':score_stuff, 'probs': prob_stuff, 'bracket_score': score, 'round_correct': round_correct, 'apply_button':1})

# @app.get("/results", response_class=HTMLResponse)
# async def get_results(request: Request):
#     session_id = await get_session_id(request)
#     session_data = await get_session_data(session_id)
#     #print('Session data on results:', session_data)
#     if ('predictions' not in session_data) or (session_data.get('round_correct') is None):
#         print((session_data.get('round_correct') is None))
#         print(('predictions' not in session_data))
#         return templates.TemplateResponse("results.html", {
#             "request": request,
#             "error": "No model type selected. Please go back and select a model type.",
#             "redirect_url": "/model_selection"
#         })    
#     return templates.TemplateResponse("results.html", {"request": request,  "prediction_method": session_data['prediction_method'], 'groups': session_data['group_stuff'], 'scores':session_data['score_stuff'], 'probs': session_data['prob_stuff'], 'bracket_score': session_data['score'], 'round_correct': session_data['round_correct'], 'apply_button':1})


from pydantic import BaseModel

class DeletePresetRequest(BaseModel):
    name: str

@app.post("/save_parameter_set")
async def save_parameter_set(request: Request, parameter_set: ParameterSet):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    if 'saved_parameter_sets' not in session_data:
        session_data['saved_parameter_sets'] = []
    session_data['saved_parameter_sets'].append(parameter_set.dict())
    session_data['selected_preset'] = parameter_set.name  # Store the selected preset
    await save_session_data(session_id, session_data)
    return {"message": "Parameter set saved successfully"}
class UpdatePresetAndParametersRequest(BaseModel):
    name: str
    parameters: List[str]

@app.post("/update_selected_preset_and_parameters")
async def update_selected_preset_and_parameters(request: Request, update_request: UpdatePresetAndParametersRequest):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    session_data['selected_preset'] = update_request.name
    session_data['selected_parameters'] = update_request.parameters
    await save_session_data(session_id, session_data)
    return {"message": "Selected preset and parameters updated successfully"}

@app.post("/delete_parameter_set")
async def delete_parameter_set(request: Request, delete_request: SelectedPreset):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    if 'saved_parameter_sets' in session_data:
        session_data['saved_parameter_sets'] = [preset for preset in session_data['saved_parameter_sets'] if preset['name'] != delete_request.name]
        if session_data.get('selected_preset') == delete_request.name:
            session_data['selected_preset'] = None  # Clear the selected preset if it was deleted
        await save_session_data(session_id, session_data)
        return {"message": "Preset deleted successfully"}
    return {"message": "Preset not found"}, 404

from called_functions.prediction_functions import *
@app.post('/applied_model')
async def applied_model(request: Request):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    encoded_model = session_data['model']
    serialized_model = base64.b64decode(encoded_model)
    model= pickle.loads(serialized_model)

    groups_stuff, predictions_dict = messing_with_predictions(model, session_data['selected_parameters'], session_data['prediction_method'], session_data['model_type'])
    if(session_data['prediction_method'] == 'score'):
        score_stuff = assign_score(predictions_dict['Score'])
    else:
        score_stuff = None

    if(session_data['model_type'] == 'random_forest'):
        prob_stuff = assign_prob(predictions_dict['probabilities'])
    else:
        prob_stuff = None
    
    return templates.TemplateResponse("results.html", {"request": request, "prediction_method": session_data['prediction_method'], 'groups': groups_stuff, 'scores':score_stuff, 'probs': prob_stuff})
    
@app.get('/applied_model', response_class=HTMLResponse)
async def get_applied_model(request: Request):
    session_id = await get_session_id(request)
    session_data = await get_session_data(session_id)
    encoded_model = session_data['model']
    serialized_model = base64.b64decode(encoded_model)
    model = pickle.loads(serialized_model)

    groups_stuff, predictions_dict = messing_with_predictions(model, session_data['selected_parameters'], session_data['prediction_method'], session_data['model_type'])
    if(session_data['prediction_method'] == 'score'):
        score_stuff = assign_score(predictions_dict['Score'])
    else:
        score_stuff = None

    if(session_data['model_type'] == 'random_forest'):
        prob_stuff = assign_prob(predictions_dict['probabilities'])
    else:
        prob_stuff = None
    
    return templates.TemplateResponse("results.html", {"request": request, "prediction_method": session_data['prediction_method'], 'groups': groups_stuff, 'scores':score_stuff, 'probs': prob_stuff})

# Add these imports at the top
# Create a connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.progress = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        self.active_connections[session_id].remove(websocket)
    
    async def broadcast_progress(self, session_id: str, progress: int, status: str = "training"):
        # Update local progress
        if session_id not in self.progress:
            self.progress[session_id] = 0
        self.progress[session_id] = progress
        
        # Publish to Redis
        message = json.dumps({"session_id": session_id, "progress": progress, "status": status})
        await redis_client.publish("progress_updates", message)
        
        # Also try to send directly if connection exists in this worker
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json({"progress": progress, "status": status})
                except Exception as e:
                    print(f"Error sending to WebSocket: {str(e)}")
    
    def get_progress(self, session_id: str):
        return self.progress.get(session_id, 0)

manager = ConnectionManager()

# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await manager.connect(websocket, session_id)
#     try:
#         # Send current progress when connecting
#         await websocket.send_json({"progress": manager.get_progress(session_id)})
#         while True:
#             # Keep connection alive
#             await websocket.receive_text()
#     except WebSocketDisconnect:
#         manager.disconnect(websocket, session_id)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    # Set up Redis subscription
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("progress_updates")
    
    # Task to listen for Redis messages
    async def listen_for_updates():
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                if data["session_id"] == session_id:
                    await websocket.send_json({"progress": data["progress"], "status": data["status"]})
    
    # Start the listener task
    listener_task = asyncio.create_task(listen_for_updates())
    
    try:
        # Send initial progress
        await websocket.send_json({"progress": manager.get_progress(session_id)})
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        listener_task.cancel()  # Cancel the listener task
        await pubsub.unsubscribe("progress_updates")

@app.post("/start_training/{session_id}")
async def start_training(session_id: str, background_tasks: BackgroundTasks):
    # Get session data
    session_data = await get_session_data(session_id)
    print(f"Starting training for session: {session_id}")
    # Start training in background
    task = asyncio.create_task(train_model_with_progress(session_id, session_data))
    print(f"Background task created: {task}")

    print(f"Added Background task: {session_id}")
    return {"status": "Training started"}

async def train_model_with_progress(session_id: str, session_data: dict):
    """Background task to train model with progress updates"""
    try:
        # Initialize progress
        await manager.broadcast_progress(session_id, 0, "Starting")
        
        # Get required data from session
        hyperparameters = session_data['hyperparameters']
        model_type = session_data['model_type']
        prediction_method = session_data['prediction_method']
        year = session_data['year']
        selected_parameters = session_data['selected_parameters']
        
        # Fix hyperparameters if needed
        hyperparameters['epochs'] = 100
        hyperparameters['learning_rate'] = 0.01
        
        # Call our wrapped version of main_distribution
        print('started training')
        predictions_dict, model = await main_distribution_with_progress(
            hyperparameters, 
            model_type, 
            prediction_method, 
            year, 
            selected_parameters,
            session_id,
            manager  # Pass the connection manager
        )
        print("finished training")
        # Process results
        serialized_model = pickle.dumps(model)
        model_encoded = base64.b64encode(serialized_model).decode('utf-8')
        groups_stuff, round_correct, score = assign_groups(predictions_dict['winners'], year)
        
        # Store results in session
        session_data['model'] = model_encoded
        session_data['predictions'] = predictions_dict
        session_data['round_correct'] = round_correct
        session_data['group_stuff'] = groups_stuff
        
        if prediction_method == 'score':
            score_stuff = assign_score(predictions_dict['Score'])
        else:
            score_stuff = None
            
        if model_type == 'random_forest':
            prob_stuff = assign_prob(predictions_dict['probabilities'])
        else:
            prob_stuff = None
        
        session_data['score_stuff'] = score_stuff
        session_data['prob_stuff'] = prob_stuff
        session_data['score'] = score
        #print(prob_stuff)
        # Save the updated session data
        await save_session_data(session_id, session_data)
        
        # Mark as complete
        await manager.broadcast_progress(session_id, 100, "Complete")
        
    except Exception as e:
        # If an error occurs, report it
        print(f"Error in model training: {e}")
        await manager.broadcast_progress(session_id, -1, f"Error: {str(e)}")

import asyncio

async def main_distribution_with_progress(hyperparameters, model_type, method, year, parameters, session_id, manager):
    # Create a container for progress updates
    progress_data = {"progress": 0, "status": "Starting"}
    
    # Create a synchronous callback
    def progress_callback(progress, status):
        progress_data["progress"] = progress
        progress_data["status"] = status
    
    # Start a background task to periodically update progress
    async def progress_monitor():
        while progress_data["progress"] < 100 and progress_data["progress"] >= 0:
            await manager.broadcast_progress(
                session_id, 
                progress_data["progress"], 
                progress_data["status"]
            )
            await asyncio.sleep(0.5)  # Update every half second
    
    # Start the progress monitor task
    monitor_task = asyncio.create_task(progress_monitor())
    
    # Run the model in a thread pool
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            partial(main_distribution, hyperparameters, model_type, method, year, parameters, progress_callback)
        )
        
        # Set progress to 100% when complete
        progress_data["progress"] = 100
        progress_data["status"] = "Complete"
        
        # Wait for the final update to be sent
        await asyncio.sleep(0.5)
        
        return result
    except Exception as e:
        # Mark as error
        progress_data["progress"] = -1
        progress_data["status"] = f"Error: {str(e)}"
        await asyncio.sleep(0.5)
        raise e
    finally:
        # Wait for monitor to finish
        await monitor_task