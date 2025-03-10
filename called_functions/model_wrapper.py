# In a new file called model_wrapper.py
import asyncio
from functools import partial
from modelfunctions import main_distribution

# Import your main_distribution and other functions
# from your_model_file import main_distribution, fix_parameters, create_teams, etc.

async def main_distribution_with_progress(hyperparameters, model_type, method, year, parameters, session_id, manager):
    """
    Wrapper around main_distribution that reports progress via WebSockets
    """
    try:
        # Report initial progress
        await manager.broadcast_progress(session_id, 5, "Starting model preparation")
        
        # Begin model training process
        await manager.broadcast_progress(session_id, 10, "Preparing data")
        
        # First phase of computation
        await manager.broadcast_progress(session_id, 20, f"Setting up {model_type} model")
        
        if method == 'winner':
            await manager.broadcast_progress(session_id, 30, "Processing win prediction data")
        elif method == 'score':
            await manager.broadcast_progress(session_id, 30, "Processing score prediction data")
        
        # Call your actual training function but check progress at intervals
        # This part runs in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        # Create a partial function that we'll run in the executor
        training_func = partial(
            main_distribution, 
            hyperparameters, 
            model_type, 
            method, 
            year, 
            parameters
        )
        print("who even knows what this is doing")
        # Run model training in a separate thread and await result
        await manager.broadcast_progress(session_id, 40, "Training model")
        output_dict, best_model = await loop.run_in_executor(None, training_func)
        print('killing myself')
        # Model training complete
        await manager.broadcast_progress(session_id, 80, "Computing final predictions")
        
        # Final processing step
        await manager.broadcast_progress(session_id, 95, "Finalizing results")

        return output_dict, best_model
        
    except Exception as e:
        # Report any errors
        await manager.broadcast_progress(session_id, -1, f"Error: {str(e)}")
        raise e