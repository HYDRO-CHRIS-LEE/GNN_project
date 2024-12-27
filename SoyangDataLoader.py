import os
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

class SoYangRiverDatasetLoader(object):
    def __init__(self, adj, data_dir=os.path.join(os.getcwd(), "Data/ffill_data"), precip_data_path='Data/rainfall/rainfall_1h_final.csv'):
        self.data_dir = data_dir
        self.precip_data_path = precip_data_path
        self.A = None
        self.X = None
        self.time = None
        self.norm = {}
        self.data_center_idx_mapping = {}

        if adj == "Euc_down":
            self._load_adjacency_matrix("Data/adj_matrix/rev2_Euclidean_Distances_downstream.csv")
        elif adj == "Euc":
            self._load_adjacency_matrix("Data/adj_matrix/rev2_Euclidean_Distances.csv")
        elif adj == "Hyd":
            self._load_adjacency_matrix("Data/adj_matrix/rev2_Hydrologic_Distances.csv")
        elif adj == "Hyd_down":
            self._load_adjacency_matrix("Data/adj_matrix/rev2_Hydrologic_Distances_downstream.csv")
        else:
            print("Wrong Adj")
            
        self._read_stored_data()
        
    def _load_adjacency_matrix(self, adj_matrix_path):
        adj_matrix = pd.read_csv(adj_matrix_path, index_col=0)
        adj_matrix = adj_matrix.fillna(0)
        self.A = adj_matrix.to_numpy()
        self.data_center_idx_mapping = {int(data_center_number): idx for idx, data_center_number in enumerate(adj_matrix.columns)}
        
    def _read_stored_data(self):
        # Load precipitation data
        precip_df = pd.read_csv(self.precip_data_path, index_col=0)
        precip_df.index = pd.to_datetime(precip_df.index)

        # Initialize containers
        dataframes = {}
        all_times_precip = set(precip_df.index)
        all_times_features = set()
    
        # Loop through each file in the data directory for other features
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                data_center_number = int(filename.split('_')[0])
                if data_center_number in self.data_center_idx_mapping:
                    file_path = os.path.join(self.data_dir, filename)
                    df = pd.read_csv(file_path, usecols=['time', 'discharge', 'water_mark_wl'])
                    df['time'] = pd.to_datetime(df['time'])
                    df['year'] = df['time'].dt.year  # Add year flag
                    df = df.set_index('time').ffill().reset_index()
                    dataframes[data_center_number] = df
                    all_times_features.update(df['time'])
    
        # Find overlapping times between features, precipitation, and radar data
        overlapping_times = all_times_features.intersection(all_times_precip)
        if not overlapping_times:
            raise ValueError("No overlapping times found between feature, and precipitation data.")
    
        # Sort and convert to list for reindexing
        overlapping_times = sorted(list(overlapping_times))
    
        # Combine other features with precipitation and radar data for overlapping times only
        for center in self.data_center_idx_mapping.keys():
            if str(center) in precip_df.columns and str(center):
                if center in dataframes:
                    center_df = dataframes[center]
                    center_df = center_df.set_index('time').reindex(overlapping_times).reset_index()
                    center_df['precipitation'] = precip_df.loc[overlapping_times, str(center)].values
                    center_df = center_df.ffill()
                    dataframes[center] = center_df
    
        # Continue with data preparation for the ML model
        self._prepare_data_for_model(dataframes, overlapping_times)
    
    def _prepare_data_for_model(self, dataframes, all_times):
        # Convert DataFrames to NumPy arrays and align
        aligned_arrays = []
        year_scaler = StandardScaler()  # Scaler for year data
        
        # Prepare a list of years for scaling
        years = np.concatenate([df['year'].values.reshape(-1, 1) for df in dataframes.values()])
        year_scaler.fit(years)  # Fit scaler on all years
        

        for center, df in sorted(dataframes.items(), key=lambda x: self.data_center_idx_mapping[x[0]]):
            # Scale the year data
            scaled_year = year_scaler.transform(df['year'].values.reshape(-1, 1)).flatten()

            # Append scaled year as a new feature
            feature_array = df[['discharge', 'precipitation', 'water_mark_wl']].to_numpy(na_value=np.nan)
            feature_array = np.hstack([feature_array, scaled_year[:, None]])  # Add year as a new column

            aligned_arrays.append(feature_array)
        
        # Stack arrays and apply standard scaling
        X = np.stack(aligned_arrays, axis=0)
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(X.shape)
        
        # Convert to Torch tensor
        self.X = torch.from_numpy(X.astype(np.float32)).permute(0, 2, 1)  # Channels last to channels first for PyTorch
        
        # Update the time attribute to match all_times sorted
        self.time = sorted(list(all_times))

        self.norm['year_scaler'] = year_scaler
        self.norm['scaler'] = scaler

    def _generate_task(self, num_timesteps_in: int = 5, num_timesteps_out: int = 1):
            """Uses the node features of the graph and generates a feature/target
            relationship of the shape
            (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
            predicting the soyang inflow using num_timesteps_in to predict the
            inflow in the next num_timesteps_out

            Args:
                num_timesteps_in (int): number of timesteps the sequence model sees
                num_timesteps_out (int): number of timesteps the sequence model has to predict
            """
            indices = [
                (i, i + (num_timesteps_in + num_timesteps_out))
                for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
            ]

            # Generate observations
            features, target = [], []
            for i, j in indices:
                features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
                target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

            self.features = features
            self.targets = target

    def _get_edges_and_weights(self):
        # Extract edges and weights from the adjacency matrix
        edge_indices = np.array(np.nonzero(self.A)).astype(np.int64)
        values = self.A[edge_indices[0], edge_indices[1]]
        self.edges = edge_indices
        self.edge_weights = values

    def get_dataset(
            self, num_timesteps_in: int = 5, num_timesteps_out: int = 1
        ) -> StaticGraphTemporalSignal:
            """Returns data iterator for soyang dataset

            Return types:
                * **dataset** *(StaticGraphTemporalSignal)* 
            """
            self._get_edges_and_weights()
            self._generate_task(num_timesteps_in, num_timesteps_out)
            dataset = StaticGraphTemporalSignal(
                self.edges, self.edge_weights, self.features, self.targets
            )
            return dataset