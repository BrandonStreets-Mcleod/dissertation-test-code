# dissertation-test-code
This repository contains all of the code required for my dissertation

# Setting up environment

1. Install Helm, Docker and Git if not already done so
2. Create a Kubernetes Cluster within Docker
3. Created sample service deployment using https://www.techtarget.com/searchitoperations/tutorial/How-to-auto-scale-Kubernetes-pods-for-microservices to create sample microservice
4. Installed Prometheus Operator using Helm and this (https://github.com/prometheus-operator/kube-prometheus) using this configuation (https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack#configuration)
5. Pip install Locust from https://locust.io/

# Starting all services

1. Run `locust -f trafficGenerator.py` to start locust generator on http://localhost:8089. Host Address is http://localhost:8080 to point to correct microservice on Docker.
2. Start Prometheus using `kubectl port-forward svc/kube-prometheus-stack-1724-prometheus 9090`
3. Enable port forwarding using `kubectl port-forward svc/microsvc 8080:8080`

# Prometheus Queries
Data and time last full run: `2024-09-11 12:21:42`<br><br>
`sum(rate(container_cpu_usage_seconds_total{pod=~"microsvc-.*"}[5m])) by (pod)` - gets the last 5 mins of cpu usage per pod<br><br>
`(sum(rate(container_cpu_usage_seconds_total{pod=~"microsvc-c76cc785b-nc7mk"}[10m])) by (pod)) / (sum(kube_pod_container_resource_limits{pod=~"microsvc-c76cc785b-nc7mk", resource="cpu"}) by (pod)) * 100` - gets percentage of CPU used by pod<br><br>
`kube_horizontalpodautoscaler_status_current_replicas` - gets the number of replicas per HPA<br><br>
`(kubelet_http_requests_duration_seconds_sum{path="metrics"}/kubelet_http_requests_duration_seconds_count{path="metrics"}) * 1000` - Gets response time in milliseconds<br><br>

# Exporting Prometheus data to CSV
run `python3 export_csv.py http://localhost:9090 <Date begin e.g. 2022-12-14T10:00:00Z> <Date end e.g. 2022-12-14T11:30:00Z> metrics.txt`

# Utilities used
 - https://github.com/hifly81/prometheus-csvplot - created code to export from Prometheus to csv

# Workload Prediction Algorithm Evolution
### 1st Iteration
Started with basic LSTM model to predict and compared with simple moving average model to set baseline.
MSE - 7.176146507263184
```
def traffic_prediction_lstm():
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, recurrent_activation='relu', input_shape=(num,1)))
    model.add(LSTM(30, return_sequences=True, recurrent_activation='relu', input_shape=(num,1)))
    model.add(LSTM(20, return_sequences=True, recurrent_activation='relu', input_shape=(num,1)))
    model.add(Dense(40,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model
```

### 2nd Iteration
What changed:
Removed Redundant input_shape:

The input_shape should only be specified in the first LSTM layer. Subsequent LSTM layers inherit the shape automatically. Removing input_shape from subsequent LSTM layers helps prevent potential issues and redundancies.
Recurrent Activation:

The recurrent_activation='relu' is not typical for LSTM cells, as they usually use sigmoid for recurrent activation. However, some variants and custom configurations might use relu. Generally, it’s best to stick with defaults unless you have a specific reason to change them.
Added Dropout:

A Dropout layer with a rate of 0.2 is added after the LSTM layers to help prevent overfitting. You can adjust the dropout rate based on your dataset and problem.
return_sequences:

In the final LSTM layer, set return_sequences=False (or omit the parameter) because you typically don’t need the full sequence output for the final dense layer, which is meant to produce a single output.
Activation Functions:

The activation functions are appropriate: relu for hidden layers and linear for the output layer. Ensure that linear is used for regression tasks.
Compiling and Summary:

Compiling the model with adam optimizer and mse loss function is correct for regression tasks. The model summary will provide a good overview of the architecture and parameters.
MSE - 3.4495973587036133
```
def traffic_prediction_lstm():
    model = Sequential()
    # First LSTM layer with return_sequences=True to pass sequences to the next LSTM layer
    model.add(LSTM(40, return_sequences=True, input_shape=(num, 1)))
    # Second LSTM layer
    model.add(LSTM(30, return_sequences=True))
    # Third LSTM layer
    model.add(LSTM(20))
    # Adding a Dropout layer to prevent overfitting
    model.add(Dropout(0.2))
    # Dense layer for additional processing
    model.add(Dense(40, activation='relu'))
    # Output layer with linear activation for regression
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Print the model summary
    model.summary()
    return model
```

### 3rd Iteration
What changed
Bidirectional LSTM:

The Bidirectional wrapper allows the model to learn from both past and future data in the sequence, which can improve performance for many time series tasks.
Increased Units and Stacked Layers:

Increased the number of units in the LSTM layers to 50, 40, and 30. Adding more units can capture more complex patterns, but it may also increase the risk of overfitting. Monitor performance and adjust as needed.
Dropout and Recurrent Dropout:

Added dropout and recurrent_dropout parameters to LSTM layers to reduce overfitting. Adjust these values based on your dataset.
Batch Normalization:

Added BatchNormalization to stabilize and accelerate training by normalizing the outputs of the LSTM layers.
Learning Rate Scheduler:

ReduceLROnPlateau adjusts the learning rate when the validation loss plateaus, which can help in fine-tuning the model and improving convergence.
MSE - 1.797590732574463
```
def traffic_prediction_lstm():
    model = Sequential()
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=(num, 1)))
    # Additional LSTM layers
    model.add(LSTM(40, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(30, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
    # Batch Normalization
    model.add(BatchNormalization())
    # Dense layer with ReLU activation
    model.add(Dense(30, activation='relu'))
    # Output layer for regression
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    # Print model summary
    model.summary()
    return model, lr_scheduler
```

### 4th Iteration
Added l2 regularisation
Played with early stopping, model checkpoint, units, all hyperparameter tuning
MSE - 0.007545528933405876
```
def traffic_prediction_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=(num, 1)))
    model.add(Bidirectional(LSTM(20, return_sequences=True, kernel_regularizer=l2(0.01))))
    model.add(Bidirectional(LSTM(10, return_sequences=False, kernel_regularizer=l2(0.01))))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    # Print model summary
    model.summary()
    return model, lr_scheduler


lstm_model, lr_scheduler = traffic_prediction_lstm()
# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```