from locust import HttpUser, LoadTestShape, TaskSet, constant, task


class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")


class WebsiteUser(HttpUser):
    wait_time = constant(0.5)
    tasks = [UserTasks]


class StagesShape(LoadTestShape):
    stop_at_end = True
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
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 300, "users": 250, "spawn_rate": 10},
        {"duration": 300, "users": 350, "spawn_rate": 10},
        {"duration": 300, "users": 200, "spawn_rate": 10},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 90, "users": 600, "spawn_rate": 100},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 300, "users": 200, "spawn_rate": 10},
        {"duration": 60, "users": 500, "spawn_rate": 100},
        {"duration": 120, "users": 250, "spawn_rate": 10},
        {"duration": 180, "users": 200, "spawn_rate": 10},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 90, "users": 500, "spawn_rate": 100},
        {"duration": 360, "users": 100, "spawn_rate": 10},
        {"duration": 360, "users": 300, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()
        stage_duration = 0
        for stage in self.stages:
            stage_duration += stage["duration"]
            if run_time < stage_duration:
                return stage["users"], stage["spawn_rate"]

        return None
