import random
from locust import HttpUser, TaskSet, LoadTestShape, constant, task

class UserTasks(TaskSet):
    @task
    def get_root(self):
        self.client.get("/employee")

class WebsiteUser(HttpUser):
    wait_time = constant(1)
    tasks = [UserTasks]

class RandomTrafficShape(LoadTestShape):
    """
    A LoadTestShape that creates random bursts of traffic and idle periods periodically over 6 hours.
    """
    
    # Total duration of the test
    total_test_duration = 6 * 60 * 60  # 6 hours in seconds

    # Maximum and minimum durations for bursts and idle periods
    max_burst_duration = 30  # 5 minutes in seconds
    min_burst_duration = 6   # 1 minute in seconds
    max_idle_duration = 30   # 5 minutes in seconds
    min_idle_duration = 6   # 1 minute in seconds
    max_normal_traffic_duration = 60  # 10 minutes in seconds
    min_normal_traffic_duration = 12  # 2 minutes in seconds

    def __init__(self):
        super().__init__()
        self.current_stage_end_time = self.get_run_time() + self._get_random_duration(self.min_burst_duration, self.max_burst_duration)
        self.current_users = 0
        self.spawn_rate = 0
        self.in_idle_period = False

    def _get_random_duration(self, min_duration, max_duration):
        return random.randint(min_duration, max_duration)
    
    def _get_random_users_and_spawn_rate(self):
        users = random.randint(50, 500)  # Random number of users
        spawn_rate = random.randint(5, 50)  # Random spawn rate
        return users, spawn_rate

    def tick(self):
        run_time = self.get_run_time()
        if (run_time <= self.total_test_duration):
            if self.in_idle_period:
                print("CHANGED TO NORMAL TRAFFIC")
                # Transition from idle to normal traffic
                self.current_users, self.spawn_rate = self._get_random_users_and_spawn_rate()
                self.current_stage_end_time = run_time + self._get_random_duration(self.min_burst_duration, self.max_burst_duration)
                self.in_idle_period = False
            else:
                if random.choice([True, False]):
                    # Randomly choose to go into idle or continue normal traffic
                    print("CHANGED TO IDLE TRAFFIC")
                    self.current_users = 0
                    self.spawn_rate = 0
                    self.current_stage_end_time = run_time + self._get_random_duration(self.min_idle_duration, self.max_idle_duration)
                    self.in_idle_period = True
                else:
                    # Continue with normal traffic
                    print("CONTINUED TO NORMAL TRAFFIC")
                    self.current_users, self.spawn_rate = self._get_random_users_and_spawn_rate()
                    self.current_stage_end_time = run_time + self._get_random_duration(self.min_normal_traffic_duration, self.max_normal_traffic_duration)

        # Return the number of users and spawn rate if they are set
        return (self.current_users, self.spawn_rate) if self.current_users or self.spawn_rate else None
