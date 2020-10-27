import numpy as np
import time
from collections import OrderedDict
import argparse
import subprocess
import string
import random
import logging
import sys


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def logging_setup(out_path=None):
    if logging.root:
        del logging.root.handlers[:]
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(out_path)),
            logging.StreamHandler(stream=sys.stdout),
        ],
        # format="[%(asctime)s:%(levelname)s:%(module)s:%(funcName)s] %(message)s",
        format="[%(asctime)s/%(levelname)s/%(module)s] %(message)s",
        datefmt="%Y-%m-%d/%H:%M",
    )


def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def format_seconds(secs_in, millis=True):
    s = []
    days, secs = divmod(secs_in, 24 * 60 * 60)
    if days > 0:
        s.append(f"{int(days)}d")
    hours, secs = divmod(secs, 60 * 60)
    if hours > 0:
        s.append(f"{int(hours):02d}h")
    mins, secs = divmod(secs, 60)
    if mins > 0:
        s.append(f"{int(mins):02d}m")
    if millis:
        s.append(f"{secs:06.3f}s")
    else:
        s.append(f"{int(secs):02d}s")
    s = "".join(s)
    return s


class Timer(object):
    def __init__(self):
        self.tic = time.time()

    def done(self):
        diff = time.time() - self.tic
        return diff

    def __call__(self):
        return self.done()

    def __str__(self):
        diff = self.done()
        return format_seconds(diff)


class StopWatch(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.timings = OrderedDict()
        self.starts = {}

    def toogle(self, name):
        if name in self.starts:
            self.stop(name)
        else:
            self.start(name)

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        tic = time.time()
        if name not in self.timings:
            self.timings[name] = []
        diff = tic - self.starts.pop(name, tic)
        self.timings[name].append(diff)
        return diff

    def get(self, name=None, reduce=np.sum):
        if name is not None:
            return reduce(self.timings[name])
        else:
            ret = {}
            for k in self.timings:
                ret[k] = reduce(self.timings[k])
            return ret

    def format_str(self, reduce=np.sum):
        return ", ".join(
            [
                f"{k}: {format_seconds(v)}"
                for k, v in self.get(reduce=reduce).items()
            ]
        )

    def __repr__(self):
        return self.format_str()

    def __str__(self):
        return self.format_str()


class ETA(object):
    def __init__(self, length, current_idx=0):
        self.reset(length, current_idx=current_idx)

    def reset(self, length=None, current_idx=0):
        if length is not None:
            self.length = length
        self.current_idx = current_idx
        self.start_time = time.time()
        self.current_time = time.time()

    def update(self, idx):
        self.current_idx = idx
        self.current_time = time.time()

    def inc(self):
        self.current_idx += 1
        self.current_time = time.time()

    def get_elapsed_time(self):
        return self.current_time - self.start_time

    def get_item_time(self):
        return self.get_elapsed_time() / (self.current_idx + 1)

    def get_remaining_time(self):
        return self.get_item_time() * (self.length - self.current_idx + 1)

    def get_total_time(self):
        return self.get_item_time() * self.length

    def get_elapsed_time_str(self, millis=True):
        return format_seconds(self.get_elapsed_time(), millis=millis)

    def get_remaining_time_str(self, millis=True):
        return format_seconds(self.get_remaining_time(), millis=millis)

    def get_percentage_str(self):
        perc = self.get_elapsed_time() / self.get_total_time() * 100
        return f"{int(perc):02d}%"

    def get_str(
        self, percentage=True, elapsed=True, remaining=True, millis=False
    ):
        s = []
        if percentage:
            s.append(self.get_percentage_str())
        if elapsed:
            s.append(self.get_elapsed_time_str(millis=millis))
        if remaining:
            s.append(self.get_remaining_time_str(millis=millis))
        return "/".join(s)


def flatten(vals):
    if isinstance(vals, dict):
        ret = []
        for v in vals.values():
            ret.extend(flatten(v))
        return ret
    elif isinstance(vals, (list, np.ndarray)):
        if isinstance(vals, np.ndarray):
            vals = vals.ravel()
        ret = []
        for v in vals:
            ret.extend(flatten(v))
        return ret
    else:
        return [vals]


class CumulativeMovingAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.vals = None

    def append(self, x):
        if isinstance(x, dict):
            if self.n == 0:
                self.vals = {}
                for k, v in x.items():
                    self.vals[k] = np.array(v)
            else:
                for k, v in x.items():
                    self.vals[k] = (np.array(v) + self.n * self.vals[k]) / (
                        self.n + 1
                    )
        else:
            x = np.asarray(x)
            if self.n == 0:
                self.vals = x
            else:
                self.vals = (x + self.n * self.vals) / (self.n + 1)
        self.n += 1
        return self.vals

    def vals_list(self):
        return flatten(self.vals)


def git_hash(cwd=None):
    ret = subprocess.run(
        ["git", "describe", "--always"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    hash = ret.stdout
    if hash is not None and "fatal" not in hash.decode():
        return hash.decode().strip()
    else:
        return None
