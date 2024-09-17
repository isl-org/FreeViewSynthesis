import numpy as np
import torch
import torch.utils.data
import random
import logging
import time
import datetime
from pathlib import Path
import argparse
import subprocess
import socket
import sys
import os
import gc

from . import utils
from . import sqlite


def log_datetime():
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def log_cuda_mem():
    logging.info(
        f"current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    logging.info(
        f"max memory allocated:     {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )
    logging.info(
        f"cached memory:            {torch.cuda.memory_cached() / 1024**2:.2f} MB"
    )


def log_tensor_memory_report():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                logging.info(f"{type(obj)}: {obj.size()}")
        except:
            pass


def get_parser(additional_commands=None):
    commands = ["retrain", "resume", "eval", "eval-init", "slurm"]
    if additional_commands:
        commands += additional_commands
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="resume", choices=commands)
    parser.add_argument("--log-env-info", type=utils.str2bool, default=False)
    parser.add_argument("--iter", type=str, nargs="*", default=[])
    parser.add_argument("--eval-net-root", type=str, default="")
    parser.add_argument("--experiments-root", type=str, default="./experiments")
    parser.add_argument("--slurm-cmd", type=str, default="resume")
    parser.add_argument("--slurm-queue", type=str, default="gpu")
    parser.add_argument("--slurm-n-gpus", type=int, default=1)
    parser.add_argument("--slurm-n-cpus", type=int, default=-1)
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="2-00:00",
        help='Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"',
    )
    return parser


class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, n_train_iters, train_iter=0):
        self.n_train_iters = n_train_iters
        self.train_iter = train_iter

    def __len__(self):
        return self.n_train_iters

    def __iter__(self):
        rng = np.random.RandomState()
        rng.seed(27644437)
        ind = rng.permutation(self.n_train_iters).tolist()
        ind = ind[self.train_iter :]
        return iter(ind)


class WorkerObjects(object):
    def __init__(
        self, *, net_f=None, optim_f=None, lr_scheduler_f=None, net_init_f=None
    ):
        self.net_f = net_f
        self.optim_f = optim_f
        self.lr_scheduler_f = lr_scheduler_f
        self.net_init_f = net_init_f

    def get_net(self):
        net = self.net_f()
        if self.net_init_f is not None:
            net.apply(self.net_init_f)
        return net

    def get_optimizer(self, net):
        return self.optim_f(net)

    def get_lr_scheduler(self, optimizer):
        return (
            None
            if self.lr_scheduler_f is None
            else self.lr_scheduler_f(optimizer)
        )


class Frequency(object):
    def __init__(self, iter=0, hours=0, minutes=0, seconds=0):
        self.freq_iter = iter
        self.freq_time_delta = datetime.timedelta(
            hours=hours, minutes=minutes, seconds=seconds
        ).total_seconds()
        self.n_resets = -1
        if self.freq_iter < 0 and self.freq_time_delta < 0:
            raise Exception("invalid Frequency, will never be True")

    def set_train_set_len(self, train_set_len):
        if self.freq_iter < 0:
            self.freq_iter = -self.freq_iter * train_set_len

    def reset(self):
        self.n_resets += 1
        self.start_time = time.time()
        self.current_iter = 0

    def advance(self):
        self.current_time = time.time()
        self.current_iter += 1
        if (self.freq_iter > 0 and self.current_iter >= self.freq_iter) or (
            self.freq_time_delta > 0
            and (self.current_time - self.start_time) > self.freq_time_delta
        ):
            self.reset()
            return True
        return False

    def get_elapsed_time(self):
        return self.current_time - self.start_time

    def get_item_time(self):
        return self.get_elapsed_time() / (self.current_iter + 1)

    def get_remaining_time(self):
        iter_time = self.get_item_time() * (
            self.freq_iter - self.current_iter + 1
        )
        time_delta_time = self.freq_time_delta - (
            self.current_time - self.start_time
        )
        if self.freq_iter > 0 and self.freq_time_delta > 0:
            return min(iter_time, time_delta_time)
        elif self.freq_iter > 0:
            return iter_time
        elif self.freq_time_delta > 0:
            return time_delta_time
        else:
            raise Exception("invalid Frequency")

    def get_total_time(self):
        iter_time = self.get_item_time() * self.freq_iter
        if self.freq_iter > 0 and self.freq_time_delta > 0:
            return min(iter_time, self.freq_time_delta)
        elif self.freq_iter > 0:
            return iter_time
        elif self.freq_time_delta > 0:
            return self.freq_time_delta
        else:
            raise Exception("invalid Frequency")

    def get_elapsed_time_str(self, millis=True):
        return utils.format_seconds(self.get_elapsed_time(), millis=millis)

    def get_remaining_time_str(self, millis=True):
        return utils.format_seconds(self.get_remaining_time(), millis=millis)

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


def dataset_rng(idx):
    rng = np.random.RandomState()
    rng.seed(idx)
    return rng


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, name, train=True, logging_rate=16):
        self.name = name
        self.current_epoch = 0
        self.train = train
        self.logging_rate = logging_rate

    def base_len(self):
        raise NotImplementedError("")

    def base_getitem(self, idx, rng):
        raise NotImplementedError("")

    def __len__(self):
        return self.base_len()

    def __getitem__(self, idx):
        rng = dataset_rng(idx)
        idx = idx % len(self)
        return self.base_getitem(idx, rng)


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, name, *datasets, uniform_sampling=False):
        self.name = name
        self.datasets = []
        self.n_samples = []
        self.cum_n_samples = [0]
        self.uniform_sampling = uniform_sampling

        for dataset in datasets:
            self.append(dataset)

    @property
    def logging_rate(self):
        return min([dset.logging_rate for dset in self.datasets])

    @logging_rate.setter
    def logging_rate(self, logging_rate):
        for dset in self.datasets:
            dset.logging_rate = logging_rate

    def append(self, dataset):
        if not isinstance(dataset, BaseDataset):
            raise Exception("invalid Dataset in append")
        self.datasets.append(dataset)
        self.n_samples.append(len(dataset))
        n_samples = self.cum_n_samples[-1] + len(dataset)
        self.cum_n_samples.append(n_samples)

    def __len__(self):
        return self.cum_n_samples[-1]

    def __getitem__(self, idx):
        rng = dataset_rng(idx)
        if self.uniform_sampling:
            didx = rng.randint(0, len(self.datasets))
            sidx = rng.randint(0, self.n_samples[didx])
        else:
            idx = idx % len(self)
            didx = np.searchsorted(self.cum_n_samples, idx, side="right") - 1
            sidx = idx - self.cum_n_samples[didx]
        return self.datasets[didx].base_getitem(sidx, rng)


class Worker(object):
    def __init__(
        self,
        experiments_root="./experiments",
        experiment_name=None,
        n_train_iters=-128,
        seed=42,
        train_batch_size=8,
        train_batch_acc_steps=1,
        eval_batch_size=16,
        num_workers=16,
        save_frequency=None,
        eval_frequency=None,
        train_device="cuda:0",
        eval_device="cuda:0",
        clip_gradient_value=None,
        clip_gradient_norm=None,
        empty_cache_per_batch=False,
        log_debug=None,
    ):
        self.experiments_root = Path(experiments_root)
        if experiment_name is None:
            experiment_name = self.exec_script_name()
        self.experiment_name = experiment_name
        self.n_train_iters = n_train_iters
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.train_batch_acc_steps = train_batch_acc_steps
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.save_frequency = (
            save_frequency if save_frequency is not None else Frequency(iter=-1)
        )
        self.eval_frequency = (
            eval_frequency if eval_frequency is not None else Frequency(iter=-1)
        )
        self.train_device = train_device
        self.eval_device = eval_device
        self.clip_gradient_value = clip_gradient_value
        self.clip_gradient_norm = clip_gradient_norm
        self.empty_cache_per_batch = empty_cache_per_batch
        # all, param, grad, grad_norm, loss, out
        self.log_debug = [] if log_debug is None else log_debug

        self.train_iter_messages = []
        self.stopwatch = utils.StopWatch()

    def exec_script_name(self):
        return os.path.splitext(os.path.basename(os.path.abspath(sys.argv[0])))[
            0
        ]

    def setup_experiment(self):
        hostname = socket.gethostname()
        self.exp_out_root = self.experiments_root / self.experiment_name
        self.exp_out_root.mkdir(parents=True, exist_ok=True)

        utils.logging_setup(
            out_path=self.exp_out_root / f"train.{hostname}.log"
        )

        logging.info(f"Set seed to {self.seed}")
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.db_path = self.exp_out_root / f"exp.{hostname}.db"
        self.db_logger = sqlite.Logger(self.db_path)
        self.db_logger.add_table(
            sqlite.Table(
                "metrics",
                fields=[
                    sqlite.StrField("method"),
                    sqlite.IntField("iter"),
                    sqlite.StrField("dataset"),
                    sqlite.StrField("metric"),
                    sqlite.FloatField("value"),
                ],
                constraints=[
                    sqlite.Constraint(
                        field_names=["method", "iter", "dataset", "metric"]
                    )
                ],
            )
        )

    def log_experiment_start(self, type="UNKNOWN", log_env_info=False):
        logging.info("=" * 80)
        logging.info(f'Start cmd "{type}": {self.experiment_name}')
        log_datetime()
        logging.info(f"host: {socket.gethostname()}")

        logging.info("-" * 80)
        env_str = []
        for k, v in self.__dict__.items():
            env_str.append(f"    {k}: {v}")
        env_str = "\n".join(env_str)
        logging.info(f"worker env:\n{env_str}")

        if log_env_info:
            from torch.utils import collect_env

            logging.info("-" * 80)
            logging.info(f"git hash of project: {utils.git_hash()}")
            logging.info(f"ENV:\n{collect_env.get_pretty_env_info()}")
            logging.info(f"Device: {torch.cuda.get_device_properties(0)}")
        logging.info("=" * 80)

    def metric_add_train(self, iter, metric, value, method=None):
        if method is None:
            method = self.experiment_name
        self.db_logger.insert(
            "metrics",
            method=method,
            iter=iter,
            dataset="train",
            metric=metric,
            value=value,
        )

    def metric_add_eval(self, iter, dataset, metric, value, method=None):
        if method is None:
            method = self.experiment_name
        self.db_logger.insert(
            "metrics",
            method=method,
            iter=iter,
            dataset="eval/" + dataset,
            metric=metric,
            value=value,
        )

    def do_cmd(self, args, worker_objects):
        self.log_experiment_start(type=args.cmd, log_env_info=args.log_env_info)

        if args.cmd == "retrain":
            self.train(worker_objects, resume=False)
        elif args.cmd == "resume":
            self.train(worker_objects, resume=True)
        elif args.cmd == "eval":
            self.eval_iters(
                worker_objects, iters=args.iter, net_root=args.eval_net_root
            )
        elif args.cmd == "eval-init":
            eval_sets = self.get_eval_sets()
            self.eval(-1, worker_objects.get_net(), eval_sets)
        elif args.cmd == "slurm":
            self.slurm(args)
        else:
            raise Exception("invalid cmd")

    def do(self, args, worker_objects):
        self.setup_experiment()
        self.do_cmd(args, worker_objects)

    def slurm(self, args):
        slurm_cmd = args.slurm_cmd

        if args.slurm_n_cpus <= 0:
            n_cpus = args.slurm_n_gpus * 12
        slurm_sh_path = (
            self.exp_out_root
            / f"{self.experiment_name}_slurm_{slurm_cmd}_{int(time.time())}.sh"
        )
        # slurm_err_path = self.exp_out_root / f"slurm%j_err.txt"
        slurm_out_path = self.exp_out_root / f"slurm%j_out.txt"
        script_path = Path(sys.argv[0]).resolve()

        def _unparse(k, v):
            if isinstance(v, list):
                return f"--{k.replace('_', '-')} {' '.join(map(str, v))}"
            else:
                return f"--{k.replace('_', '-')} {str(v)}"

        slurm_args = vars(args)
        slurm_args["cmd"] = slurm_cmd
        slurm_args["log_env_info"] = "1"
        slurm_args = [
            _unparse(k, v)
            for k, v in slurm_args.items()
            if not (
                "slurm" in k
                or v is None
                or (isinstance(v, str) and len(v) == 0)
                or (isinstance(v, list) and len(v) == 0)
            )
        ]
        slurm_args = " ".join(slurm_args)

        def _write(fp, txt):
            logging.info(txt)
            fp.write(f"{txt}\n")

        with open(slurm_sh_path, "w") as fp:
            _write(fp, f"#!/bin/bash")
            _write(fp, f"#SBATCH --partition {args.slurm_queue}")
            _write(fp, f"#SBATCH --gres=gpu:{args.slurm_n_gpus}")
            _write(fp, f"#SBATCH --cpus-per-task {n_cpus}")
            _write(fp, f"#SBATCH --time {args.slurm_time}")
            # _write(fp, f"#SBATCH --error {slurm_err_path}")
            _write(fp, f"#SBATCH --output {slurm_out_path}")
            _write(fp, f"python -u {script_path} {slurm_args}")
        cmd = ["sbatch", str(slurm_sh_path)]
        logging.info(" ".join(cmd))
        ret = subprocess.run(cmd)
        logging.info(ret.stdout)
        logging.info(ret)

    def get_train_set(self):
        # returns train_set
        raise NotImplementedError()

    def get_eval_sets(self):
        # returns eval_sets
        raise NotImplementedError()

    def copy_data(self, data, device, train):
        # self.im = data['im'].to(self.device).requires_grad_(requires_grad=train)
        raise NotImplementedError()

    def free_copied_data(self):
        pass

    def net_forward(self, net, train, iter):
        raise NotImplementedError()

    def loss_forward(self, output, train, iter):
        raise NotImplementedError()

    def callback_train_post_backward(self, **kwargs):
        # net = kwargs['net']
        # err = False
        # for name, param in net.named_parameters():
        #   if not torch.isfinite(param.grad).all():
        #     print(f'{name} has non-finite gradient')
        #     err = True
        # if err:
        #   __import__('ipdb').set_trace()
        pass

    def callback_eval_start(self, **kwargs):
        pass

    def callback_eval_add(self, **kwargs):
        pass

    def callback_eval_stop(self, **kwargs):
        pass

    def get_train_data_loader(self, dset, iter):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=self.train_batch_size,
            shuffle=False,
            sampler=TrainSampler(
                self.n_train_iters * self.train_batch_size, train_iter=iter
            ),
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def get_eval_data_loader(self, dset):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def format_err_str(self, errs, div=1):
        err_list = []
        for v in errs.values():
            if isinstance(v, np.ndarray):
                err_list.extend(v.ravel())
            elif isinstance(v, list):
                v=np.array(v)
                err_list.extend(v.ravel())
            else:
                err_list.append(v)
        err = sum(err_list)
        if len(err_list) > 1:
            err_str = f"{err/div:0.4f}=" + "+".join(
                [f"{e/div:0.4f}" for e in err_list]
            )
        else:
            err_str = f"{err/div:0.4f}"
        return err_str

    def get_net_path(self, iter, net_root=None):
        if net_root is None:
            net_root = self.exp_out_root
        return net_root / f"net_{iter:016d}.params"

    def get_net_paths(self, net_root=None):
        if net_root is None:
            net_root = self.exp_out_root
        net_paths = {}
        for net_path in sorted(net_root.glob("net_*.params")):
            iter = int(net_path.with_suffix("").name[4:])
            net_paths[str(iter)] = (net_path, iter)
            net_paths["last"] = (net_path, iter)
        return net_paths

    def eval_iters(self, worker_objects, iters=None, net_root=None):
        if net_root is None or net_root == "":
            net_root = self.exp_out_root
        else:
            net_root = Path(net_root)
        net_paths = self.get_net_paths(net_root=net_root)

        if iters is None or len(iters) == 0:
            iters = [iter for iter in net_paths.keys() if iter != "last"]

        eval_sets = self.get_eval_sets()
        net = worker_objects.get_net()

        if "init" in iters:
            self.eval(-1, net, eval_sets)
            iters.remove("init")

        for iter_str in iters:
            if iter_str in net_paths:
                net_path, iter = net_paths[iter_str]
                logging.info(
                    f"[EVAL] loading net for iter {iter_str}: {net_path}"
                )
                state_dict = torch.load(
                    str(net_path), map_location=self.eval_device
                )
                net.load_state_dict(state_dict)
                self.eval(iter, net, eval_sets)
            else:
                logging.info(f"[EVAL] no network params for iter {iter_str}")

    def eval(self, iter, net, eval_sets, epoch="x"):
        for eval_set_idx, eval_set in enumerate(eval_sets):
            logging.info("")
            logging.info("=" * 80)
            logging.info(f"Evaluating set {eval_set.name}")
            self.eval_set(iter, net, eval_set_idx, eval_set, epoch=epoch)

    def eval_set(self, iter, net, eval_set_idx, eval_set, epoch="x"):
        torch.cuda.empty_cache()
        with torch.no_grad():
            logging.info("-" * 80)
            log_datetime()
            logging.info("Eval iter %d" % iter)
            eval_loader = self.get_eval_data_loader(eval_set)

            net = net.to(self.eval_device)
            net.eval()

            mean_loss = utils.CumulativeMovingAverage()
            self.stopwatch.reset()

            self.stopwatch.start("callback")
            self.callback_eval_start(
                iter=iter, net=net, set_idx=eval_set_idx, eval_set=eval_set
            )
            self.stopwatch.stop("callback")

            # torch.cuda.empty_cache()
            # logging.info("--------- init ----------")
            # log_cuda_mem()
            # log_tensor_memory_report()
            # logging.info("-------------------------")

            eta = utils.ETA(length=len(eval_loader))
            self.stopwatch.start("total")
            self.stopwatch.start("data")
            for batch_idx, data in enumerate(eval_loader):
                # if batch_idx == 4:
                #     break

                if self.empty_cache_per_batch:
                    torch.cuda.empty_cache()

                self.copy_data(data, device=self.eval_device, train=False)
                # logging.info("--------- copy data ----------")
                # log_cuda_mem()
                self.stopwatch.stop("data")

                self.stopwatch.start("forward")
                output = self.net_forward(net, train=False, iter=iter)
                if "cuda" in self.eval_device:
                    torch.cuda.synchronize()
                # logging.info("--------- forward ----------")
                # log_cuda_mem()
                self.stopwatch.stop("forward")

                self.stopwatch.start("loss")
                errs = self.loss_forward(output, train=False, iter=iter)
                err_items = {}
                for k in errs.keys():
                    if torch.is_tensor(errs[k]):
                        err_items[k] = errs[k].item()
                    else:
                        err_items[k] = [v.item() for v in errs[k]]
                del errs
                mean_loss.append(err_items)
                # logging.info("--------- loss ----------")
                # log_cuda_mem()
                self.stopwatch.stop("loss")

                eta.update(batch_idx)
                if batch_idx % eval_set.logging_rate == 0:
                    err_str = self.format_err_str(err_items)
                    logging.info(
                        f"eval {epoch}/{iter}: {batch_idx+1}/{len(eval_loader)}: loss={err_str} ({np.sum(mean_loss.vals_list()):0.4f}) | {eta.get_str(percentage=True, elapsed=True, remaining=True)}"
                    )

                self.stopwatch.start("callback")
                self.callback_eval_add(
                    iter=iter,
                    net=net,
                    set_idx=eval_set_idx,
                    eval_set=eval_set,
                    batch_idx=batch_idx,
                    n_batches=len(eval_loader),
                    output=output,
                )
                self.stopwatch.stop("callback")

                self.free_copied_data()
                # logging.info("--------- end ----------")
                # log_cuda_mem()

                self.stopwatch.start("data")
            self.stopwatch.stop("total")

            self.stopwatch.start("callback")
            self.callback_eval_stop(
                iter=iter,
                net=net,
                set_idx=eval_set_idx,
                eval_set=eval_set,
                mean_loss=mean_loss.vals,
            )
            self.stopwatch.stop("callback")

            logging.info("timings: %s" % self.stopwatch)

            err_str = self.format_err_str(mean_loss.vals)
            logging.info(f"avg eval_loss={err_str}")
            self.db_logger.commit()

    def train(self, worker_objects, resume=False):
        train_set = self.get_train_set()
        eval_sets = self.get_eval_sets()

        # get worker objects
        net = worker_objects.get_net()
        net = net.to(self.train_device)
        optimizer = worker_objects.get_optimizer(net)
        lr_scheduler = worker_objects.get_lr_scheduler(optimizer)

        # laod state if existent
        iter = 0
        state_path = self.exp_out_root / "state.dict"
        if resume and state_path.exists():
            logging.info("=" * 80)
            logging.info(f"Loading state from {state_path}")
            logging.info("=" * 80)
            state = torch.load(str(state_path), map_location=self.train_device)
            iter = state["iter"] + 1
            net.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng_state"].to("cpu"))
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state["gpu_rng_state"].to("cpu"))

        # update lr_scheduler
        if lr_scheduler is not None:
            old_lr = lr_scheduler.get_lr()
            for _ in range(iter):
                lr_scheduler.step()
            new_lr = lr_scheduler.get_lr()
            if old_lr != new_lr:
                logging.info(
                    f"(RESUME) Update LR {old_lr} => {new_lr} via lr_scheduler iter={iter}"
                )

        # compute n_train_iters based on number of samples in train set
        if self.n_train_iters < 0:
            self.n_train_iters = -self.n_train_iters * len(train_set)

        # set-up training variables
        train_loader = self.get_train_data_loader(train_set, iter)
        iter_range = list(range(iter, self.n_train_iters))
        eta_total = utils.ETA(length=len(iter_range))
        mean_loss = utils.CumulativeMovingAverage()
        net.train()
        optimizer.zero_grad()

        # init frequencies
        self.save_frequency.set_train_set_len(len(train_set))
        self.eval_frequency.set_train_set_len(len(train_set))
        self.save_frequency.reset()
        self.eval_frequency.reset()

        self.stopwatch.reset()
        self.stopwatch.start("total")
        self.stopwatch.start("data")
        for iter, data in zip(iter_range, train_loader):
            # set-up
            self.train_iter_messages = [f"train {iter+1}/{self.n_train_iters}"]

            # copy data
            self.copy_data(data, device=self.train_device, train=True)
            self.stopwatch.stop("data")

            # forward pass
            self.stopwatch.start("forward")
            output = self.net_forward(net, train=True, iter=iter)
            if "cuda" in self.train_device:
                torch.cuda.synchronize()
            self.stopwatch.stop("forward")

            # evaluate loss, convert loss values to scalars
            self.stopwatch.start("loss")
            errs = self.loss_forward(output, train=True, iter=iter)
            self.free_copied_data()
            try:
                err = sum(
                    [v if torch.is_tensor(v) else sum(v) for v in errs.values()]
                )
                err_items = {}
                for k in errs.keys():
                    if torch.is_tensor(errs[k]):
                        err_items[k] = errs[k].item()
                    else:
                        err_items[k] = [v.item() for v in errs[k]]
                del errs
                mean_loss.append(err_items)
            except TypeError as type_error:
                self.train_iter_messages.append(
                    f"No loss computed due to TypeError: {type_error}"
                )
                eta_total.inc()
                self.train_iter_messages.append(
                    f"{self.save_frequency.n_resets}/{self.save_frequency.get_str(percentage=True, elapsed=False, remaining=True)}"
                )
                self.train_iter_messages.append(
                    f"{eta_total.get_str(percentage=True, elapsed=True, remaining=True)}"
                )
                logging.info(" | ".join(self.train_iter_messages))
                continue
            if "cuda" in self.train_device:
                torch.cuda.synchronize()
            self.stopwatch.stop("loss")

            # backward pass
            self.stopwatch.start("backward")
            if self.train_batch_acc_steps > 1:
                err = err / self.train_batch_acc_steps
            err.backward()
            self.callback_train_post_backward(
                net=net, errs=err_items, output=output, iter=iter
            )
            if "cuda" in self.train_device:
                torch.cuda.synchronize()
            self.stopwatch.stop("backward")

            # optimizer step
            optimizer_steped = False
            if (iter + 1) % self.train_batch_acc_steps == 0:
                self.stopwatch.start("optimizer")
                if self.clip_gradient_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        net.parameters(), self.clip_gradient_value
                    )
                if self.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), self.clip_gradient_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steped = True
                if "cuda" in self.train_device:
                    torch.cuda.synchronize()
                self.stopwatch.stop("optimizer")

            # evaluate frequencies
            do_save = self.save_frequency.advance()
            do_eval = self.eval_frequency.advance()

            # show progress
            eta_total.inc()
            if iter < 128 or iter % train_set.logging_rate == 0:
                err_str = self.format_err_str(err_items)
                self.train_iter_messages.append(
                    f"loss={err_str} ({'y' if optimizer_steped else 'n'}{np.sum(mean_loss.vals_list()):0.4f})"
                )
                self.train_iter_messages.append(
                    f"{self.save_frequency.n_resets}/{self.save_frequency.get_str(percentage=True, elapsed=False, remaining=True)}"
                )
                self.train_iter_messages.append(
                    f"{eta_total.get_str(percentage=True, elapsed=True, remaining=True)}"
                )
                logging.info(" | ".join(self.train_iter_messages))

            self.stopwatch.stop("total")

            # save state and network params
            if do_save or iter == iter_range[-1]:
                # store network
                net_path = self.get_net_path(iter)
                logging.info("-" * 80)
                logging.info(f"save network to {net_path}")
                torch.save(net.state_dict(), str(net_path))

                # store state
                state_dict = {
                    "iter": iter,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    state_dict["gpu_rng_state"] = torch.cuda.get_rng_state()
                state_tmp_path = self.exp_out_root / "state.dict.tmp"
                logging.info(f"save state to {state_tmp_path}")
                torch.save(state_dict, str(state_tmp_path))
                logging.info(f"rename {state_tmp_path} to {state_path}")
                state_tmp_path.rename(state_path)

                # log avergae train loss and  average timing
                self.metric_add_train(
                    iter, "loss", np.sum(mean_loss.vals_list())
                )
                err_str = self.format_err_str(mean_loss.vals)
                mean_loss.reset()
                logging.info("-" * 80)
                logging.info(f"avg train_loss={err_str}")
                logging.info(f"timings: {self.stopwatch}")
                logging.info("=" * 80)
                self.stopwatch.reset()

                # commit logger
                self.db_logger.commit()

            # eval network
            if do_eval:
                self.eval(
                    iter, net, eval_sets, epoch=self.save_frequency.n_resets
                )
                net = net.to(self.train_device)
                net.train()
                logging.info("")
                logging.info("=" * 80)

            # update lr
            if lr_scheduler is not None:
                old_lr = lr_scheduler.get_lr()
                lr_scheduler.step()
                new_lr = lr_scheduler.get_lr()
                if old_lr != new_lr:
                    logging.info(
                        f"Update LR {old_lr} => {new_lr} via lr_scheduler iter={iter}"
                    )

            self.stopwatch.start("total")
            self.stopwatch.start("data")
            # end of iter loop

        logging.info("=" * 80)
        logging.info("Finished training")
        log_datetime()
        logging.info("=" * 80)
