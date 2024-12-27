import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

import networkx as nx
from sklearn.metrics import mean_squared_error, r2_score
import hydroeval as he

def calculate_metrics(predictions, actuals):
    """
    Calculate performance metrics for a given set of predictions and actual values.
    """
    # Calculate R-squared
    r_squared = r2_score(actuals, predictions)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # Calculate PBIAS
    pbias = 100 * (np.sum(actuals - predictions) / np.sum(actuals))
    
    # Calculate NSE
    nse = he.evaluator(he.nse, predictions, actuals)[0]
    
    # Calculate KGE
    kge = he.evaluator(he.kge, predictions, actuals)[0]

    # Filter out zero and negative values from both actual and predicted flows before percentile calculation
    valid_indices = (actuals > 0) & (predictions > 0)
    valid_actuals = actuals[valid_indices]
    valid_predictions = predictions[valid_indices]

    # Calculate PbiasFHV: top 2% high flow values, after screening for valid values
    if len(valid_actuals) > 0:
        threshold_high = np.percentile(valid_actuals, 98)
        actual_high_flows = valid_actuals[valid_actuals > threshold_high]
        predicted_high_flows = valid_predictions[valid_actuals > threshold_high]
        pbias_fhv = 100 * (np.sum(predicted_high_flows - actual_high_flows) / np.sum(actual_high_flows))
    else:
        pbias_fhv = np.nan  # Handle the case where no valid data points are available

    # Calculate PbiasFLV: bottom 30% low flow range, after screening for valid values
    if len(valid_actuals) > 0:
        threshold_low = np.percentile(valid_actuals, 30)
        actual_low_flows = valid_actuals[valid_actuals < threshold_low]
        predicted_low_flows = valid_predictions[valid_actuals < threshold_low]

        # Only consider positive values for log calculation
        valid_low_indices = (actual_low_flows > 0) & (predicted_low_flows > 0)
        actual_low_flows = actual_low_flows[valid_low_indices]
        predicted_low_flows = predicted_low_flows[valid_low_indices]

        # Only compute PbiasFLV if there are valid non-zero and non-negative values
        if len(actual_low_flows) > 0:
            pbias_flv = 100 * (np.sum(np.log(predicted_low_flows) - np.log(actual_low_flows)) / np.sum(np.log(actual_low_flows)))
        else:
            pbias_flv = np.nan  # Handle the case where no valid data points are available
    else:
        pbias_flv = np.nan  # Handle the case where no valid data points are available



    return r_squared, rmse, pbias, nse, kge, pbias_fhv, pbias_flv



class visualization:
    def __init__(self, loader, predictions, test_labels, figsave_path, figsave_casename, loss_fn):
        self.loader = loader
        self.predictions = predictions
        self.test_labels = test_labels
        self.figsave_path = figsave_path
        self.figsave_casename = figsave_casename
        self.loss_fn = loss_fn
        self.metrics_df = None
        self.predictions_unscaled, self.labels_unscaled, self.station_predictions, self.station_labels, self.idx_to_station = self.prepare_data()

    def prepare_data(self):
        # Assuming that the 'loader' object and the corresponding scalers are available
        scaler = self.loader.norm['scaler']
        year_scaler = self.loader.norm['year_scaler']

        # Concatenate predictions and labels
        predictions_concat = np.concatenate([pred[:, :, 0].detach().cpu().numpy() for pred in self.predictions], axis=0)
        labels_concat = np.concatenate([lbl[:, :, 0].detach().cpu().numpy() for lbl in self.test_labels], axis=0)

        # Ensure the reshaping matches the number of features used in the scaler
        num_nodes = predictions_concat.shape[1]
        num_features = scaler.mean_.shape[0]  # number of features used in scaling

        # Reshape to match the original scaling shape (samples * nodes, features)
        predictions_reshaped = predictions_concat.reshape(-1, num_features)
        labels_reshaped = labels_concat.reshape(-1, num_features)

        # Inverse transform to undo scaling
        predictions_unscaled = scaler.inverse_transform(predictions_reshaped)
        labels_unscaled = scaler.inverse_transform(labels_reshaped)

        # Reshape back to the desired shape (samples, nodes)
        predictions_unscaled = predictions_unscaled.reshape(predictions_concat.shape[0], num_nodes)
        labels_unscaled = labels_unscaled.reshape(labels_concat.shape[0], num_nodes)

        # Now, split by stations
        station_predictions = [predictions_unscaled[:, i] for i in range(predictions_unscaled.shape[1])]
        station_labels = [labels_unscaled[:, i] for i in range(labels_unscaled.shape[1])]

        # Assuming data_center_idx_mapping is available here
        # Reverse the mapping to get the station number by index
        data_center_idx_mapping = self.loader.data_center_idx_mapping
        idx_to_station = {idx: station for station, idx in data_center_idx_mapping.items()}

        return predictions_unscaled, labels_unscaled, station_predictions, station_labels, idx_to_station
 

    def scattor_plot(self):
        print("Making scatter plot for each station")
        station_predictions = self.station_predictions
        station_labels = self.station_labels
        idx_to_station = self.idx_to_station

        # Calculate MSE for each station
        station_mse = [self.loss_fn(torch.tensor(station_predictions[i]), torch.tensor(station_labels[i])).item() for i in range(len(station_predictions))]

        # Determine the number of rows and columns for the subplots
        num_stations = len(station_predictions)
        num_cols = 3
        num_rows = (num_stations + num_cols - 1) // num_cols

        # Create a figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12.8, 4*num_rows), dpi=300)

        # Plot scatter plot for each station
        for i, ax in enumerate(axes.flat):
            if i < num_stations:
                ax.scatter(station_labels[i], station_predictions[i], color='#096cc2', alpha=0.5)
                ax.plot([min(station_labels[i].min(), station_predictions[i].min()), max(station_labels[i].max(), station_predictions[i].max())],
                        [min(station_labels[i].min(), station_predictions[i].min()), max(station_labels[i].max(), station_predictions[i].max())],
                        color='red', linestyle='--')
                ax.set_xlabel('True Discharge [m³/s]')
                ax.set_ylabel('Predicted Discharge [m³/s]')
                ax.set_title(f'Station {idx_to_station[i]} (MSE: {station_mse[i]:.4f})')
            else:
                ax.axis('off')

        # Adjust spacing between subplots
        plt.tight_layout()
        plt.savefig(f"{self.figsave_path}ScatterPlot_{self.figsave_casename}.png")

    def time_series(self):
        print("Making Time series visualization data...")
        predictions_unscaled = self.predictions_unscaled
        idx_to_station = self.idx_to_station
        labels_unscaled = self.labels_unscaled
        
        # Extract time points for the plot
        time_points = self.loader.time[-predictions_unscaled.shape[0]:]

        # Load precipitation data
        precip_df = pd.read_csv(self.loader.precip_data_path, index_col=0)
        precip_df.index = pd.to_datetime(precip_df.index)

        # Prepare the subplot environment
        n_stations = predictions_unscaled.shape[1]
        fig, axs = plt.subplots(n_stations, 1, figsize=(15, 5 * n_stations))

        if n_stations == 1:
            axs = [axs]

        # Plot predictions and actual values for each station
        for i, ax in enumerate(axs):
            station_id = idx_to_station[i]  # Get the station number using idx_to_station
            # Create primary axis for discharge
            ax1 = ax
            ax1.plot(time_points, predictions_unscaled[:, i], label='Predictions', color='#0a6cff', linewidth=1.5, zorder=2, alpha=0.8)
            ax1.plot(time_points, labels_unscaled[:, i], label='Actual', color='black', linewidth=2, zorder=1)
            
            ax1.set_title(f'Station {station_id} Predictions vs Actual with Precipitation')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Discharge [m³/s]')
            ax1.legend(loc='upper left')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.minorticks_on()

            # Set y-axis limits for discharge with 50% padding at the top
            max_discharge = max(predictions_unscaled[:, i].max(), labels_unscaled[:, i].max())
            ax1.set_ylim(-1, max_discharge * 1.5)

            # Create secondary axis for precipitation
            ax2 = ax1.twinx()
            
            # Extract precipitation data for the current station and time range
            station_precip = precip_df.loc[time_points, str(station_id)]
            
            # Plot precipitation as upside-down bar chart
            ax2.bar(time_points, station_precip, color='grey', alpha=0.3, width=0.2)
            ax2.set_ylabel('Precipitation [mm/hr]')
            ax2.invert_yaxis()  # Invert the y-axis for precipitation
            
            # Set reasonable limits for precipitation axis
            max_precip = station_precip.max()
            ax2.set_ylim(max_precip * 1.75, 0)  # Add 75% padding to the top

        # Format the x-axis to handle datetime objects
        axs[0].xaxis.set_major_locator(MaxNLocator(10))
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.figsave_path}TimeSeries_{self.figsave_casename}.png", dpi=300)

    # def metrics(self):
    #     print("Calculating metrics: R^2, RMSE, PBIAS, NSE, KGE")
    #     predictions_unscaled = self.predictions_unscaled
    #     idx_to_station = self.idx_to_station
    #     labels_unscaled = self.labels_unscaled

    #     # Prepare a DataFrame to store the results
    #     metrics_df = pd.DataFrame(columns=['Station', 'R-squared', 'RMSE', 'PBIAS', 'NSE', 'KGE'])

    #     # Loop through each station to calculate the metrics
    #     for i in range(predictions_unscaled.shape[1]):
    #         station_id = str(idx_to_station[i])  # Get the station number using idx_to_station
            
    #         # Get the predictions and actual values for the current station
    #         predictions = predictions_unscaled[:, i]
    #         actuals = labels_unscaled[:, i]
            
    #         # Calculate the metrics
    #         r_squared, rmse, pbias, nse, kge = calculate_metrics(predictions, actuals)
            
    #         # Append the results to the DataFrame
    #         metrics_df = metrics_df.append({
    #             'Station': station_id,
    #             'R-squared': r_squared,
    #             'RMSE': rmse,
    #             'PBIAS': pbias,
    #             'NSE': nse,
    #             'KGE': kge[0]
    #         }, ignore_index=True)

    #     # Display the results
    #     print(metrics_df)
    #     csv_file_path = f"{self.figsave_path}Metric_{self.figsave_casename}.csv"
    #     metrics_df.to_csv(csv_file_path, index=False)

    #     self.metrics_df = metrics_df

    def metrics(self):
        print("Calculating metrics: R^2, RMSE, PBIAS, NSE, KGE, PbiasFHV, PbiasFLV")
        predictions_unscaled = self.predictions_unscaled
        idx_to_station = self.idx_to_station
        labels_unscaled = self.labels_unscaled

        # Collect metric dictionaries in a list
        metrics_list = []

        # Loop through each station to calculate the metrics
        for i in range(predictions_unscaled.shape[1]):
            station_id = str(self.idx_to_station[i])  # Get the station number using idx_to_station

            # Get the predictions and actual values for the current station
            predictions = predictions_unscaled[:, i]
            actuals = labels_unscaled[:, i]

            # Calculate the metrics
            r_squared, rmse, pbias, nse, kge, pbias_fhv, pbias_flv = calculate_metrics(predictions, actuals)

            # Handle cases where kge is an array
            kge_value = kge[0] if isinstance(kge, np.ndarray) else kge

            # Append the results to the list
            metrics_list.append({
                'Station': station_id,
                'R-squared': r_squared,
                'RMSE': rmse,
                'PBIAS': pbias,
                'NSE': nse,
                'KGE': kge_value,
                'PbiasFHV': pbias_fhv,
                'PbiasFLV': pbias_flv
            })

        # Create the DataFrame from the list
        metrics_df = pd.DataFrame(metrics_list)

        # Display the results
        print(metrics_df)
        csv_file_path = f"{self.figsave_path}Metric_{self.figsave_casename}.csv"
        metrics_df.to_csv(csv_file_path, index=False)

        self.metrics_df = metrics_df

        # Return the metrics DataFrame
        return metrics_df



    def plot_metric_bar(self, metric):
        """
        Plots a bar plot for the given metric.
        """
        df = self.metrics_df

        # Extract values for the given metric
        metric_values = df[metric].values
        station_indices = df['Station'].values
        
        # Plot the bar plot
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        ax.bar(range(len(metric_values)), metric_values)
        ax.set_xlabel('Station Index')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} for Each Station')
        ax.set_xticks(range(len(metric_values)))
        ax.set_xticklabels(station_indices)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.figsave_path}BarPlot_{metric}_{self.figsave_casename}.png")

    def plot_metric_network(self, metric, correlations):
        """
        Plots a network plot for the given metric.

        Parameters:
            df (pd.DataFrame): DataFrame containing the metrics for each station.
            metric (str): The name of the metric to plot.
            correlations (dict): Dictionary defining the correlations between stations.
            idx_to_station (dict): Mapping from index to station ID.
            figsave_path (str): Path to save the figure.
            figsave_casename (str): Case name for the figure file.
        """
        df = self.metrics_df

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges to the graph
        for src, dst in correlations.items():
            G.add_edge(src, dst)

        # Assign metric values to each station
        station_indices = df['Station'].values
        metric_values = df[metric].values
        metric_dict = {str(station): metric for station, metric in zip(station_indices, metric_values)}

        # Define bounds and extensions for each metric (now includes PbiasFHV and PbiasFLV)
        metric_bounds = {
            'KGE': {'vmin': -2, 'vmax': 1, 'extend': 'min'},
            'NSE': {'vmin': -2, 'vmax': 1, 'extend': 'min'},
            'R-squared': {'vmin': 0, 'vmax': 1, 'extend': 'min'},
            'RMSE': {'vmin': 0, 'vmax': 10, 'extend': 'max'},
            'PBIAS': {'vmin': -50, 'vmax': 50, 'extend': 'both'},
            'PbiasFHV': {'vmin': -100, 'vmax': 100, 'extend': 'both'},  # Added bounds for PbiasFHV
            'PbiasFLV': {'vmin': -100, 'vmax': 100, 'extend': 'both'}   # Added bounds for PbiasFLV
        }

        # Get bounds for the current metric
        vmin = metric_bounds[metric]['vmin']
        vmax = metric_bounds[metric]['vmax']
        extend = metric_bounds[metric]['extend']

        # Normalize metric values for color mapping
        metric_normalized = [max(vmin, min(vmax, metric_dict[node])) for node in G.nodes if node not in ['outlet1', 'outlet2', 'outlet3', 'outlet4']]
        metric_color_map = plt.cm.GnBu

        # Normalize metric values for node size mapping
        metric_size_normalized = [(metric - vmin) / (vmax - vmin) for metric in metric_normalized]

        # Draw the graph
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        pos_adjusted = {
            '5002650': (-2.5, 4), '5002643': (-3, 9),
            '5002660': (-3, 1), '5002677': (-6, 0.4), '5002690': (-2, -6.36),
            '5001625': (1, 4.25), '5001640': (3, 0.72),
            '5001645': (1, 0.4), '5001660': (0.36, -2.89), '5001655': (-0.5, 0.87),
            '5001670': (7, -4), '5001673': (5, -3), '5001650': (3.5, -3.43),
            '5001627': (4.5, 3), '5101675': (4, -14), '5302620': (-7, 3.6),
            '4003650': (6, 10), '4006680': (7.5, 4), '4008670': (6, 0),
            '4009610': (9, 0), '4009630': (12, 0.3), '4009670': (20, -4),
            'outlet1': (-0.18, -10.0), 'outlet2': (20, -8), 'outlet3': (2, -17), 'outlet4': (-10, 4)
        }

        node_colors = ['lightgray' if node in ['outlet1', 'outlet2', 'outlet3', 'outlet4'] else 
                        metric_color_map((max(vmin, min(vmax, metric_dict[node])) - vmin) / (vmax - vmin)) 
                        for node in G.nodes]
        node_sizes = [150 if node in ['outlet1', 'outlet2', 'outlet3', 'outlet4'] else 
                        1000 * ((max(vmin, min(vmax, metric_dict[node])) - vmin) / (vmax - vmin)) 
                        for node in G.nodes]
        
        # Draw edges
        nx.draw_networkx_edges(G, pos_adjusted, edge_color='black', width=1, ax=ax)

        # Draw nodes with border
        border_size = [size + 50 for size in node_sizes]  # Adjust border size as needed
        nx.draw_networkx_nodes(G, pos_adjusted, node_size=border_size, node_color='gray', ax=ax)  # Border

        # Draw actual nodes on top
        nx.draw_networkx_nodes(G, pos_adjusted, node_size=node_sizes, node_color=node_colors, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos_adjusted, font_size=8, ax=ax)

        # Create a colorbar legend with specific bounds and extension
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=metric_color_map, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, extend=extend)
        cbar.set_label(metric)

        plt.title(f"{metric} Network {self.figsave_casename}")
        plt.savefig(f"{self.figsave_path}Network_{metric}_{self.figsave_casename}.png", dpi=300)


    def plot_bar(self): 
        print("Making plot bar...")
        # Define correlations (as provided in your previous code)
        correlations = {
            "5002643": "5002650",
            "5002650": "5002660",
            "5002660": "5002690",
            "5001625": "5001640",
            "5001640": "5001660",
            "5001655": "5001660",
            "5001670": "5001673",
            "5001673": "5001650",
            "5001650": "outlet1",
            "5002677": "5002690",
            "5002690": "outlet1",
            "5001660": "outlet1",
            "4003650": "4006680",
            "4008670": "4009610",
            "4006680": "4009610",
            "4009610": "4009630",
            "4009630": "outlet2",
            "4009670": "outlet2",
            "5101675": "outlet3",
            "5302620": "outlet4"
        }

        # Plot bar plots for each metric (include the new metrics)
        metrics = ['R-squared', 'RMSE', 'PBIAS', 'NSE', 'KGE', 'PbiasFHV', 'PbiasFLV']
        
        # Loop through all metrics including the new ones
        for metric in metrics:
            self.plot_metric_bar(metric)

        # Plot network plots for each metric
        for metric in metrics:
            self.plot_metric_network(metric, correlations)
