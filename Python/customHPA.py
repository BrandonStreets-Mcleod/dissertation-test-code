import time
from kubernetes import client, config
import requests
import json
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.optimizers import *
from tensorflow import keras
from sklearn.metrics import *
from datetime import datetime, timedelta
from sklearn.model_selection import *
from scipy import interpolate
import json
from pathlib import Path
from pandas import to_numeric
from hmmlearn import hmm
from keras.losses import *
from dateutil import parser
import os
from sklearn.feature_selection import VarianceThreshold

os.environ["OMP_NUM_THREADS"] = "1"  # Set the environment variable

def convert_datetime_format(datetime_str):
    target_format = "%d/%m/%Y %H:%M"
    try:
        datetime_obj = parser.parse(datetime_str)
        return datetime_obj.strftime(target_format)
    except (ValueError, TypeError):
        print(f"Invalid datetime format: {datetime_str}")
        return None

def insert_missing_minutes(group):
    group = group.sort_values(by='datetime').reset_index(drop=True)
    all_rows = []
    for i in range(len(group) - 1):
        curr_time = group.iloc[i]['datetime']
        next_time = group.iloc[i + 1]['datetime']
        while next_time - curr_time > pd.Timedelta(minutes=1):
            curr_time = curr_time + pd.Timedelta(minutes=1)
            # Insert a new row for the missing minute with NaN for 'value'
            new_row = pd.DataFrame({
                'datetime': [curr_time],
                'machine_id': [group.iloc[i]['machine_id']],
                'cpu_usage': [np.nan]
            })
            all_rows.append(new_row)
    if all_rows:
        group = pd.concat([group] + all_rows).sort_values(by='datetime').reset_index(drop=True)
    return group

def create_time_features(data):
    # Assume 'cpu_data' is your dataframe and 'datetime' is the column with datetime values
    data['datetime'] = pd.to_datetime(data['datetime'])  # Ensure it's a datetime type
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['day_of_month'] = data['datetime'].dt.day
    data['is_weekend'] = data['datetime'].dt.weekday >= 5  # True for Saturday and Sunday
    return data

# Define a function to create lagged features
def create_additional_features(df, target_col, lags):
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
    except:
        df['datetime'] = df['datetime'].apply(convert_datetime_format)
    df['cpu_usage'].replace(0, np.nan, inplace=True)
    if 'machine_id' in df.columns:
        df['machine_id'] = df['machine_id'].fillna(method='bfill')
        df['machine_id'] = pd.Categorical(df['machine_id']).codes + 1
    else:
        df['machine_id'] = 1
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
    df = df.groupby('machine_id', group_keys=False).apply(insert_missing_minutes)
    df['cpu_usage'] = df.groupby('machine_id', group_keys=False)['cpu_usage'].apply(lambda x: x.interpolate(method='linear', limit_direction='both'))
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    df['moving_avg'] = df['cpu_usage'].rolling(window=10).mean() # 10-minute moving average
    # Drop any rows with NaN values that were created due to shifting
    df.dropna(inplace=True)
    column_to_move = df.pop('cpu_usage')
    df.insert(1, 'cpu_usage', column_to_move)
    df = create_time_features(df)
    all_features_no_timestamp = df.columns[1:len(df.columns)-1]
    features_for_hmm = df[all_features_no_timestamp].values
    num_states = 10
    model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=1000, init_params="")
    scaler = StandardScaler()
    features_for_hmm_scaled = scaler.fit_transform(features_for_hmm)
    selector = VarianceThreshold(threshold=1e-5)
    features_for_hmm_scaled_reduced = selector.fit_transform(features_for_hmm_scaled)
    cov_matrix = np.cov(features_for_hmm_scaled_reduced, rowvar=False) + 1e-6 * np.eye(features_for_hmm_scaled_reduced.shape[1])
    model.covars_ = np.tile(cov_matrix, (num_states, 1, 1))
    model.fit(features_for_hmm_scaled_reduced)
    for i, row in enumerate(model.transmat_):
        if row.sum() == 0:
            model.transmat_[i] = np.full(model.n_components, 1 / model.n_components)
    hidden_states = model.predict(features_for_hmm_scaled_reduced)
    df['hidden_state'] = hidden_states
    return df

def get_time_range():
    end_time = int(time.time())  # Current timestamp in seconds
    start_time = end_time - (3 * 3600) # Hours in seconds
    return start_time, end_time

# Prometheus server URL (adjust according to your setup)
PROMETHEUS_URL = "http://localhost:9090"

def query_prometheus_range(query, start_time, end_time, step):
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step  # Interval between points (e.g., 60s = 1 minute intervals)
    }
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
    return response.json()

# Function to get CPU usage for a pod over the past hour
def get_pod_cpu_usage_range(pod_name, start_time, end_time):
    # Prometheus query to get CPU usage over the last hour (rate over 1 minutes)
    query = f'avg((sum(rate(container_cpu_usage_seconds_total{{pod=~"{pod_name}-.*"}}[1m])) by (pod)) / (sum(kube_pod_container_resource_requests{{pod=~"{pod_name}-.*", resource="cpu"}}) by (pod))) * 100'
    # Query Prometheus with a 60-second step interval
    result = query_prometheus_range(query, start_time, end_time, step="60")

    return result

# Function to convert the Prometheus result to a pandas DataFrame
def prometheus_to_dataframe(prometheus_result):
    if not prometheus_result or 'data' not in prometheus_result:
        return pd.DataFrame()  # Return an empty DataFrame if there's no data

    # Extract the 'values' from the first result (assuming only one pod)
    values = prometheus_result['data']['result'][0]['values']

    # Create a DataFrame from the values
    df = pd.DataFrame(values, columns=['datetime', 'cpu_usage'])

    # Convert UNIX timestamp to human-readable datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

    # Convert the value column to float
    df['cpu_usage'] = df['cpu_usage'].astype(float)

    return df

def get_dataset():
    start_time, end_time = get_time_range()
    pod_cpu_data = get_pod_cpu_usage_range("microsvc", start_time, end_time)
    cluster_data_1h = prometheus_to_dataframe(pod_cpu_data)
    cluster_data_1h['datetime'] = pd.to_datetime(cluster_data_1h['datetime']).dt.floor('T')
    lags = [5, 10, 60]
    full_1h_data = create_additional_features(cluster_data_1h, 'cpu_usage', lags)
    full_1h_data.dropna(inplace=True)
    return full_1h_data

window_size = 60
prediction_horizon = 10

def create_lstm_sequences(group_data, window_size, prediction_horizon):
    X, y = [], []
    for i in range(len(group_data) - window_size - prediction_horizon + 1):
        # Features are all columns except 'datetime', 'cpu_usage', and 'machine_id'
        X.append(group_data.iloc[i:i + window_size].drop(columns=['datetime', 'cpu_usage', 'machine_id']).values)
        # Target is the next 10 CPU usage values
        y.append(group_data.iloc[i + window_size:i + window_size + prediction_horizon]['cpu_usage'].values)
    return np.array(X), np.array(y)

# Load kube config
config.load_kube_config()
# If running inside the cluster, use:
# config.load_incluster_config()

def scale_out(additional_replicas, deployment_name, namespace='default'):
    api = client.AppsV1Api()
    deployment = api.read_namespaced_deployment(deployment_name, namespace)
    current_replicas = deployment.spec.replicas
    new_replicas = current_replicas + additional_replicas
    deployment.spec.replicas = new_replicas
    api.patch_namespaced_deployment(deployment_name, namespace, deployment)
    print(f"Scaled out to {new_replicas} replicas.")
    return new_replicas

def scale_in(remove_replicas, deployment_name, namespace='default'):
    api = client.AppsV1Api()
    deployment = api.read_namespaced_deployment(deployment_name, namespace)
    current_replicas = deployment.spec.replicas
    new_replicas = max(current_replicas - remove_replicas, 1)
    deployment.spec.replicas = new_replicas
    api.patch_namespaced_deployment(deployment_name, namespace, deployment)
    print(f"Scaled in to {new_replicas} replicas.")
    return new_replicas

def detect_burst(monitoring_interval, window_size, resource_prediction_model, replica_prediction_model, replicas):
    is_burst = False
    replicas_before_burst = 1
    burst_threshold = 2
    non_burst_threshold = 1.5
    past_predictions = np.array([])
    past_predictions_cpu_usage = np.array([])
    feature_scaler_burst = MinMaxScaler()
    target_scaler_burst = MinMaxScaler()
    sd_max = 0
    n_max = 0
    while True:
        time.sleep(monitoring_interval)
        full_1h_data = get_dataset()
        all_columns_no_datetime_cpu = full_1h_data.columns[2:len(full_1h_data.columns)]
        full_1h_data[all_columns_no_datetime_cpu] = feature_scaler_burst.fit_transform(full_1h_data[all_columns_no_datetime_cpu])
        full_1h_data[['cpu_usage']] = target_scaler_burst.fit_transform(full_1h_data[['cpu_usage']])
        # In 10 minute increments
        prediction_steps = 10
        prediction_lookup = 60
        predict_data, predict_labels = create_lstm_sequences(full_1h_data, prediction_lookup, prediction_steps)
        cpu_predictions = resource_prediction_model.predict(predict_data, verbose=0)
        predictions_rescaled = target_scaler_burst.inverse_transform(cpu_predictions)
        n_predicted = replica_prediction_model.predict(predictions_rescaled.reshape(-1, 1))
        past_predictions = np.append(past_predictions, n_predicted[-1])
        past_predictions_cpu_usage = np.append(past_predictions_cpu_usage, predictions_rescaled.reshape(-1, 1)[-1])
        for i in range(1, window_size + 1):
            sigma_i = np.std(past_predictions_cpu_usage[-10:])

            if sigma_i > sd_max:
                sd_max = sigma_i
                n_max = max(past_predictions[-10:].flatten())
        print(f"CPU Usage Prediction: {predictions_rescaled.reshape(-1, 1)[-1][0]}")
        print(f"Replica Prediction: {n_predicted[-1]}")
        print(f"STD of CPU Usage: {sigma_i}")
        print(f"Burst Number of replicas: {n_max}")
        if sd_max >= burst_threshold and not is_burst:
            # Detected burst, increase replicas to n_max
            replicas_during_burst = n_max
            is_burst = True
            replicas_before_burst = n_predicted[-1]  # Store current predicted replicas before burst
        elif sd_max >= burst_threshold and is_burst:
            # Continuation of the burst
            replicas_during_burst = n_max
        elif sd_max < non_burst_threshold and is_burst:
            if replicas_before_burst > n_predicted[-1]:
                # Burst ending, scale back to predicted replicas
                replicas_during_burst = n_predicted[-1]
                is_burst = False
                replicas_before_burst = 1
            else:
                # Keep scaling at n_max
                replicas_during_burst = n_max
        else:
            if predictions_rescaled.reshape(-1, 1)[-1][0] >= 70 and replicas + 1 > n_predicted[-1]:
                replicas_during_burst = replicas + 1
            else:
                replicas_during_burst = n_predicted[-1]

        current_replica_count = int(round(replicas))
        replicas_during_burst = int(round(replicas_during_burst))

        if current_replica_count < replicas_during_burst:
            additional_replicas = int(replicas_during_burst - current_replica_count)
            replicas = scale_out(additional_replicas, 'microsvc', namespace='default')
        elif current_replica_count > replicas_during_burst:
            replica_difference = int(current_replica_count - replicas_during_burst)
            replicas = scale_in(replica_difference, 'microsvc', namespace='default')
        else:
            print("No replica update needed")
        # Log the updated replica count
        print(f"Updated replica count: {replicas}")


api = client.AppsV1Api()
deployment = api.read_namespaced_deployment('microsvc', namespace='default')
replicas = deployment.spec.replicas
print(f"Original replica count: {replicas}")
lstm_model = load_model('best_model_idea.keras')
best_model = joblib.load('decision_tree_model.pkl')
detect_burst(60, 10, lstm_model, best_model, replicas)