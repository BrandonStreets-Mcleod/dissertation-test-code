from locust import HttpUser, LoadTestShape, TaskSet, constant, task


class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")


class WebsiteUser(HttpUser):
    wait_time = constant(0.5)
    tasks = [UserTasks]


class StagesShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.

    Keyword arguments:

        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage

        stop_at_end -- Can be set to stop once all stages have run.
    """

    stages = [
        {"duration": 300, "users": 1000, "spawn_rate": 100},
        {"duration": 600, "users": 2500, "spawn_rate": 100},
        {"duration": 360, "users": 3500, "spawn_rate": 100},
        {"duration": 300, "users": 2000, "spawn_rate": 100},
        {"duration": 360, "users": 1000, "spawn_rate": 100},
        {"duration": 300, "users": 500, "spawn_rate": 10},
        {"duration": 300, "users": 2500, "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None