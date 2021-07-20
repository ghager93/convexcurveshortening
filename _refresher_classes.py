import time


class ECSFRefresher:
    def start(self):
        pass

    def next_step(self):
        pass

    def is_time_to_refresh(self):
        pass

    def perform_refreshing(self, concavity: float, area_percent: float):
        print(f"Concavity: {concavity: .2f}, Area to original %: {area_percent: .2f}")

class IterativeECSFRefresher(ECSFRefresher):
    def __init__(self, refresh_iterative_interval: int):
        self.refresh_iterative_interval = refresh_iterative_interval
        self.curr_interval = 0

    def start(self):
        self.curr_interval = 0

    def next_step(self):
        self.curr_interval += 1

    def is_time_to_refresh(self):
        return not (self.curr_interval % self.refresh_iterative_interval)


class TimeECSFRefresher(ECSFRefresher):
    def __init__(self, refresh_time_interval: float):
        self.refresh_time_interval = refresh_time_interval
        self.start_time = time.time()
        self.curr_time = time.time()

    def start(self):
        self.start_time = time.time()
        self.curr_time = time.time()

    def next_step(self):
        self.curr_time = time.time()

    def is_time_to_refresh(self):
        return not ((self.curr_time - self.start_time) % self.refresh_time_interval)
