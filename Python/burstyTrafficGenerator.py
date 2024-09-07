from locust import HttpUser, TaskSet, LoadTestShape, constant, task

class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")

class WebsiteUser(HttpUser):
    wait_time = constant(1)
    tasks = [UserTasks]

class BurstsShape(LoadTestShape):
    """
    A LoadTestShape that creates bursts of traffic periodically over 6 hours.
    """

    # Define the burst stages
    stages = [
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 180, "users": 600, "spawn_rate": 100},
        {"duration": 300, "users": 100, "spawn_rate": 10},
        {"duration": 420, "users": 200, "spawn_rate": 10},
    ]

    # Total duration of the test
    total_test_duration = 6 * 60 * 60  # 6 hours in seconds
    # Total duration of the stages
    cycle_duration = sum(stage["duration"] for stage in stages)

    def tick(self):
        run_time = self.get_run_time()
        
        # Calculate the time within the current cycle
        time_in_cycle = run_time % self.cycle_duration
        
        for stage in self.stages:
            stage_end_time = stage["duration"]
            if time_in_cycle < stage_end_time:
                return (stage["users"], stage["spawn_rate"])
            time_in_cycle -= stage_end_time
        
        return None
