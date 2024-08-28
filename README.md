# dissertation-test-code
This repository contains all of the code required for my dissertation

# Starting all services

1. Run `locust -f trafficGenerator.py` to start locust generator on http://localhost:8089. Host Address is http://localhost:8080 to point to correct microservice on Docker.
2. Start Prometheus using `prometheus --config.file=prometheus.yml` in the Prometheus exe directory
3. Enable port forwarding using `kubectl port-forward svc/microsvc 8080:8080` and `kubectl port-forward svc/kube-state-metrics 
-n kube-system 8081:8081` 
4. Once all basic setup has been done, start the locust generator
