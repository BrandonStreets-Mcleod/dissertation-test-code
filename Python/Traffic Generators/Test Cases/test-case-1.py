##################
# NORMAL TRAFFIC #
##################

from locust import HttpUser, LoadTestShape, TaskSet, constant, task
from locust.contrib.fasthttp import FastHttpUser

class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")


class WebsiteUser(FastHttpUser):
    wait_time = constant(0.5)
    tasks = [UserTasks]


class StagesShape(LoadTestShape):
    stop_at_end = True

    stages = [
        {"duration": 600, "users": 100, "spawn_rate": 10},
        {"duration": 600, "users": 250, "spawn_rate": 50},
        {"duration": 420, "users": 500, "spawn_rate": 50},
        {"duration": 360, "users": 750, "spawn_rate": 100},
        {"duration": 420, "users": 500, "spawn_rate": 50},
        {"duration": 600, "users": 250, "spawn_rate": 50},
        {"duration": 600, "users": 100, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()
        stage_duration = 0
        for stage in self.stages:
            stage_duration += stage["duration"]
            if run_time < stage_duration:
                return stage["users"], stage["spawn_rate"]

        return None
