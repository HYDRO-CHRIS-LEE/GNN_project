import pandas as pd
import numpy as np

# def save_model_predictions(predictions, actuals, station_mapping, adjmethod, save_path, model_name):
#     """
#     Save predictions and actual values for each station to CSV files.
    
#     Parameters:
#     -----------
#     predictions : numpy.ndarray
#         Model predictions with shape (timesteps, stations)
#     actuals : numpy.ndarray
#         Actual values with shape (timesteps, stations)
#     station_mapping : dict
#         Mapping of station indices to station IDs
#     save_path : str
#         Path to save the CSV files
#     model_name : str
#         Name of the model (e.g., 'EALSTM', 'GWNET', 'ASTGCN')
#     """
#     # Convert indices to station IDs
#     idx_to_station = {idx: station for station, idx in station_mapping.items()}
    
#     # Validate dimensions
#     if predictions.ndim != 2 or actuals.ndim != 2:
#         raise ValueError(f"Predictions and actuals must have 2 dimensions (timesteps, stations). "
#                          f"Got predictions with shape {predictions.shape} and actuals with shape {actuals.shape}.")
    
#     if predictions.shape != actuals.shape:
#         raise ValueError(f"Predictions and actuals must have the same shape. "
#                          f"Got predictions with shape {predictions.shape} and actuals with shape {actuals.shape}.")

#     # Save data for each station
#     for i in range(predictions.shape[1]):
#         station_id = idx_to_station.get(i, f"unknown_station_{i}")
        
#         # Ensure data is 1-dimensional for each station
#         station_predictions = predictions[:, i]
#         station_actuals = actuals[:, i]

#         if station_predictions.ndim != 1 or station_actuals.ndim != 1:
#             raise ValueError(f"Station {station_id}: Data must be 1-dimensional for predictions and actuals. "
#                              f"Got prediction shape {station_predictions.shape} and actual shape {station_actuals.shape}.")
        
#         # Create DataFrame with predictions and actuals
#         df = pd.DataFrame({
#             'actual': station_actuals,
#             'prediction': station_predictions
#         })
        
#         # Save to CSV
#         filename = f"{save_path}/station_{station_id}_{model_name}_{adjmethod}_predictions.csv"
#         df.to_csv(filename, index=False)
#         print(f"Saved predictions for station {station_id} to {filename}")

def save_model_predictions(predictions, actuals, station_mapping, adjmethod, save_path, model_name, input_seq_len=None, blocks=None, layers=None):
    """
    Save predictions and actual values for each station to CSV files.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions with shape (timesteps, stations)
    actuals : numpy.ndarray
        Actual values with shape (timesteps, stations)
    station_mapping : dict
        Mapping of station indices to station IDs
    save_path : str
        Path to save the CSV files
    model_name : str
        Name of the model (e.g., 'EALSTM', 'GWNET', 'ASTGCN')
    input_seq_len : int, optional
        Input sequence length for all models (default: None)
    blocks : int, optional
        Number of blocks for GWNET (default: None)
    layers : int, optional
        Number of layers for GWNET (default: None)
    """
    # Convert indices to station IDs
    idx_to_station = {idx: station for station, idx in station_mapping.items()}
    
    # Validate dimensions
    if predictions.ndim != 2 or actuals.ndim != 2:
        raise ValueError(f"Predictions and actuals must have 2 dimensions (timesteps, stations). "
                         f"Got predictions with shape {predictions.shape} and actuals with shape {actuals.shape}.")
    
    if predictions.shape != actuals.shape:
        raise ValueError(f"Predictions and actuals must have the same shape. "
                         f"Got predictions with shape {predictions.shape} and actuals with shape {actuals.shape}.")

    # Create filename suffix based on model name
    if model_name == "GWNET":
        additional_info = f"Input_{input_seq_len}_blocks{blocks}_layers{layers}"
    else:
        additional_info = f"Input_{input_seq_len}"

    # Save data for each station
    for i in range(predictions.shape[1]):
        station_id = idx_to_station.get(i, f"unknown_station_{i}")
        
        # Ensure data is 1-dimensional for each station
        station_predictions = predictions[:, i]
        station_actuals = actuals[:, i]

        if station_predictions.ndim != 1 or station_actuals.ndim != 1:
            raise ValueError(f"Station {station_id}: Data must be 1-dimensional for predictions and actuals. "
                             f"Got prediction shape {station_predictions.shape} and actual shape {station_actuals.shape}.")
        
        # Create DataFrame with predictions and actuals
        df = pd.DataFrame({
            'actual': station_actuals,
            'prediction': station_predictions
        })
        
        # Save to CSV
        filename = f"{save_path}/station_{station_id}_{model_name}_{adjmethod}_{additional_info}_predictions.csv"
        df.to_csv(filename, index=False)
        print(f"Saved predictions for station {station_id} to {filename}")

