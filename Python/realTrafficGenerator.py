from locust import HttpUser, task, between
from locust import LoadTestShape
from datetime import datetime, time
import random

class EmployeeUser(HttpUser):
    # Initial wait time (will be adjusted dynamically later)
    wait_time = between(1, 3)

    @task
    def get_employee(self):
        self.client.get("/employee")

    def weight_based_on_time(self):
        """
        Adjust the traffic intensity based on the time of day to simulate traffic spikes.
        """
        current_time = datetime.now().time()

        # Define time periods for peak hours
        morning_peak_start = time(7, 30)
        morning_peak_end = time(9, 30)
        evening_peak_start = time(16, 30)
        evening_peak_end = time(18, 30)

        if (morning_peak_start <= current_time <= morning_peak_end) or \
           (evening_peak_start <= current_time <= evening_peak_end):
            # Peak hours: more traffic (70-90% load)
            return random.uniform(0.7, 0.9)
        else:
            # Off-peak hours: less traffic (10-30% load)
            return random.uniform(0.1, 0.3)

    def wait_time(self):
        """
        Adjust the wait time dynamically based on traffic intensity.
        During peak hours, the wait time is shorter to simulate more requests.
        """
        weight = self.weight_based_on_time()

        # Shorter wait time during peak traffic
        if weight > 0.7:
            return between(0.5, 1)  # Faster requests during peak
        else:
            return between(3, 5)  # Slower requests during off-peak


class TimeBasedLoadShape(LoadTestShape):
    """
    A load test shape that changes the user count based on the time of day.
    The number of users fluctuates with peak hours at 7:30-9:30 AM and 4:30-6:30 PM.
    """

    def tick(self):
        current_time = datetime.now().time()

        # Define time periods for peak hours
        morning_peak_start = time(7, 30)
        morning_peak_end = time(9, 30)
        evening_peak_start = time(16, 30)
        evening_peak_end = time(18, 30)

        # Define number of users during peak and non-peak hours
        if (morning_peak_start <= current_time <= morning_peak_end) or \
           (evening_peak_start <= current_time <= evening_peak_end):
            # Peak hours: 100-150 users
            return (random.randint(100, 150), random.randint(5, 10))  # (user count, spawn rate)
        else:
            # Off-peak hours: 10-30 users
            return (random.randint(10, 30), random.randint(2, 5))  # (user count, spawn rate)


class WebsiteUser(EmployeeUser):
    # WebsiteUser uses the EmployeeUser class for request simulation
    pass
