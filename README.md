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
Removed Redundant input_shape

The recurrent_activation='relu' removed

Added Dropout

In the final LSTM layer, set return_sequences=False (or omit the parameter) because you typically don’t need the full sequence output for the final dense layer, which is meant to produce a single output.

The activation functions are appropriate: relu for hidden layers and linear for the output layer. Ensure that linear is used for regression tasks.
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
Bidirectional LSTM

Increased Units and Stacked Layers

Dropout and Recurrent Dropout

Added BatchNormalization to stabilize and accelerate training by normalizing the outputs of the LSTM layers.

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

### 5th Iteration
Updated the dataset used to allow for more features to be used and simplified to ensure overfitting was avoided

MSE - 0.007719747722148895
```
def traffic_prediction_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(10, return_sequences=True, recurrent_regularizer=l2(0.01)), input_shape=(window_size, len(features))))
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
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```

### 6th Iteration
Massively simplified the model as overfitting was a major issue
MSE: 0.000014745726148291
```
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(prediction_steps))  # Outputting 10 future values

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Checkppoint to ensure the best model is used - decided on best val_loss
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
```

### 7th Iteration
Made use of CNN-LSTM to allow for patterns to be captured by CNN and then temporal dependencies to be captured by LSTM
MSE: 3.546651451
```
lstm_model = Sequential()
# Convolutional layer to capture local patterns
lstm_model.add(Conv1D(filters=120, kernel_size=2, activation='relu', input_shape=(X_train_all.shape[1], X_train_all.shape[2])))
lstm_model.add(Conv1D(filters=60, kernel_size=2, activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=2))
lstm_model.add(LSTM(units=250, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=100, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(prediction_horizon))

# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
```

### 8th Iteration
Added `shuffle=False` to fit to preserve time series ordinality as shuffling means that patterns based off time are lost
MSE: 3.580630511104647

#### Slight Update
Used HalvingRandomSearchCV to run multiple different variations in a single run and get the best result of the 10 tests. Created layer for inputs rather than adding as part of first CNN layer
```
lstm_model = Sequential()

# CNN layers
lstm_model.add(Conv1D(filters=120, kernel_size=3, activation='relu', input_shape=(X_train_all.shape[1], X_train_all.shape[2])))
lstm_model.add(BatchNormalization())  # Normalize activations
lstm_model.add(Conv1D(filters=60, kernel_size=3, activation='relu'))
lstm_model.add(BatchNormalization())  # Normalize activations
lstm_model.add(MaxPooling1D(pool_size=2))

# LSTM layers
lstm_model.add(LSTM(units=250, return_sequences=True))
lstm_model.add(LSTM(units=100, return_sequences=True))
lstm_model.add(Dropout(0.3))
lstm_model.add(LSTM(units=50, return_sequences=False))

# Output layer (multi-step prediction)
lstm_model.add(Dense(prediction_horizon))

# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', save_best_only=True, mode='min')
# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
lstm_model.compile(optimizer="adam", loss='huber_loss')  # Use Huber loss to reduce outliers' impact
lstm_model.summary()
# Train the model
history = lstm_model.fit(X_train_all, y_train_all, epochs=50, batch_size=64, validation_data=(X_test_all, y_test_all),
                         verbose=1, callbacks=[early_stopping, model_checkpoint, reduce_lr], shuffle=False)
```

### 9th Iteration
Made use of BayeSearch to find optimal hyperparameters for model but found no improvement from the previous models MSE
```
# Build the model function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train_all.shape[1], X_train_all.shape[2])))

    # CNN layers
    model.add(Conv1D(filters=hp.Int('filters_1', min_value=150, max_value=200, step=10), kernel_size=hp.Int('kernel_size', 3, 5), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=hp.Int('filters_2', min_value=50, max_value=100, step=10), kernel_size=hp.Int('kernel_size', 3, 5), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # LSTM layers
    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=150, max_value=300, step=50), return_sequences=True, kernel_regularizer=l2(1e-4)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.05)))
    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=80, max_value=120, step=10), return_sequences=True, kernel_regularizer=l2(1e-4)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.05)))
    model.add(LSTM(units=hp.Int('lstm_units_3', min_value=20, max_value=60, step=10), return_sequences=False, kernel_regularizer=l2(1e-4)))

    # Output layer
    model.add(Dense(prediction_horizon, kernel_regularizer=l2(1e-4)))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-3, sampling='LOG')), loss=Huber())

    return model

# Keras Tuner setup
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='CNN-LSTM-tuning',
    project_name='cpu_usage_tuning',
    overwrite=True
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/kaggle/working/best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Perform hyperparameter tuning
tuner.search(X_train_all, y_train_all, epochs=40, validation_data=(X_test_all, y_test_all), callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Retrieve the best model and its hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best Hyperparameters: {best_hyperparameters.values}")
```

### Final Model (best_model_idea.keras)
Testing: RMSE: 4.226888656616211, MSE: 17.866588592529297
```
# Build model
lstm_model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(100, 3, activation='relu'),
    BatchNormalization(),
    Conv1D(50, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    LSTM(150, return_sequences=True),
    Dropout(0.4),
    LSTM(100, return_sequences=True),
    Dropout(0.4),
    LSTM(50, return_sequences=False),
    Dense(prediction_horizon)
])

# Compile and train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_idea.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

lstm_model.compile(optimizer="adam", loss=Huber())
lstm_model.summary()

history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val),
                         verbose=1, callbacks=[early_stopping, model_checkpoint, reduce_lr])
```