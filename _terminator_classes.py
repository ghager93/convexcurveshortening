import time
from abc import ABCMeta, abstractmethod


class TerminatorInterface(metaclass=ABCMeta):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def next_step(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass


class IterativeECSFTerminator(TerminatorInterface):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.curr_iterations = 0

    def start(self):
        self.curr_iterations = 0

    def next_step(self):
        self.curr_iterations += 1

    def is_finished(self):
        return self.curr_iterations >= self.max_iterations


class TimeECSFTerminator(TerminatorInterface):
    def __init__(self, max_time: float):
        self.max_time = max_time
        self.start_time = time.time()
        self.curr_time = time.time()

    def start(self):
        self.start_time = time.time()
        self.curr_time = time.time()

    def is_finished(self):
        return time.time() - self.start_time >= self.max_time


class ConditionalECSFTerminator(TerminatorInterface):
    def __init__(self, concavity_threshold: float):
        self.concavity_threshold = concavity_threshold
        self.curr_concavity = concavity_threshold + 1

    def start(self, curr_concavity):
        self.curr_concavity = curr_concavity

    def next_step(self, curr_concavity):
        self.curr_concavity = curr_concavity

    def is_finished(self):
        return self.curr_concavity <= self.concavity_threshold