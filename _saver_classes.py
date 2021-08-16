import time
from abc import ABCMeta, abstractmethod


class SaverInterface(metaclass=ABCMeta):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def next_step(self):
        pass

    @abstractmethod
    def is_time_to_save(self):
        pass


class IterativeECSFSaver(SaverInterface):
    def __init__(self, save_iterative_interval: int):
        self.save_iterative_interval = save_iterative_interval
        self.curr_interval = 0

    def start(self):
        self.curr_interval = 0

    def next_step(self):
        self.curr_interval += 1

    def is_time_to_save(self):
        return not (self.curr_interval % self.save_iterative_interval)


class TimeECSFSaver(SaverInterface):
    def __init__(self, save_time_interval: float):
        self.save_time_interval = save_time_interval
        self.start_time = time.time()
        self.curr_time = time.time()

    def start(self):
        self.start_time = time.time()
        self.curr_time = time.time()

    def next_step(self):
        self.curr_time = time.time()

    def is_time_to_save(self):
        return not ((self.curr_time - self.start_time) % self.save_time_interval)
