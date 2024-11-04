#########################################
# NORMAL TRAFFIC WITH SPACED OUT SPIKES #
#########################################

from locust import HttpUser, TaskSet, LoadTestShape, constant, task
from locust.contrib.fasthttp import FastHttpUser
import time

is_burst = False

class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")

class WebsiteUser(FastHttpUser):
    if is_burst:
        wait_time = constant(0)  # Short wait time during bursts
    else:
        wait_time = constant(1)  # Normal wait time during non-burst phases
    tasks = [UserTasks]

class BurstsShape(LoadTestShape):
    stop_at_end = True

    # Define the burst stages
    stages = [
        {"duration": 600, "users": 250, "spawn_rate": 50, "burst": False},  # Normal traffic
        {"duration": 120, "users": 1000, "spawn_rate": 500, "burst": True},  # Burst traffic
        {"duration": 480, "users": 400, "spawn_rate": 200, "burst": False},  # Normal traffic
        {"duration": 300, "users": 1000, "spawn_rate": 500, "burst": True},  # Burst traffic
        {"duration": 900, "users": 500, "spawn_rate": 250, "burst": False},  # Normal traffic
        {"duration": 60, "users": 1000, "spawn_rate": 1000, "burst": True},  # Burst traffic
        {"duration": 120, "users": 600, "spawn_rate": 100, "burst": False},  # Normal traffic
        {"duration": 120, "users": 1000, "spawn_rate": 1000, "burst": True},  # Burst traffic
        {"duration": 900, "users": 300, "spawn_rate": 50, "burst": False},   # Normal traffic
    ]

    # Calculate total test duration based on stages
    cycle_duration = sum(stage["duration"] for stage in stages)

    def tick(self):
        global is_burst  # Access the global burst control variable
        run_time = self.get_run_time()

        # Stop the test if the run time exceeds the cycle duration
        if run_time >= self.cycle_duration:
            return None

        # Progress through stages based on elapsed run time
        time_in_cycle = run_time
        for stage in self.stages:
            if time_in_cycle < stage["duration"]:
                is_burst = stage["burst"]
                return (stage["users"], stage["spawn_rate"])
            time_in_cycle -= stage["duration"]

        return None