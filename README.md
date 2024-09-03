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
2. Start Prometheus using `../Prometheus/prometheus --config.file=prometheus.yml`
3. Enable port forwarding using `kubectl port-forward svc/microsvc 8080:8080`

# Prometheus Queries
`sum(rate(container_cpu_usage_seconds_total{pod=~"microsvc-.*"}[5m])) by (pod)` - gets the last 5 mins of cpu usage per pod

# Exporting Prometheus data to CSV
run `python3 export_csv.py http://localhost:9090 <Date begin e.g. 2022-12-14T10:00:00Z> <Date end e.g. 2022-12-14T11:30:00Z> metrics.txt`

# Utilities used
 - https://github.com/hifly81/prometheus-csvplot - created code to export from Prometheus to csv