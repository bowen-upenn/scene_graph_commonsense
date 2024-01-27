# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        custom_scheduler=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.custom_scheduler = custom_scheduler

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and not self.custom_scheduler:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, update_schedule=False, load_mapping={}):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_mapping)
        if with_optim:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                if update_schedule:
                    self.scheduler.last_epoch = checkpoint["iteration"]
                else:
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, load_mapping):
        load_state_dict(self.model, checkpoint.pop("model"), load_mapping)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        custom_scheduler=False,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger, custom_scheduler
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


def clip_grad_norm(named_parameters, max_norm, logger, clip=False, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

    if verbose:
        logger.info('---Total norm {:.5f} clip coef {:.5f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            logger.info("{:<50s}: {:.5f}, ({})".format(name, norm, param_to_shape[name]))
        logger.info('-------------------------------')

    return total_norm

class EBMCheckpointer(object):
    def __init__(
        self,
        cfg,
        base_model,
        energy_model,
        base_optimizer=None,
        energy_optimizer=None,
        base_scheduler=None,
        energy_scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        custom_scheduler=False,
    ):
        self.cfg = cfg.clone()

        self.base_model = base_model
        self.energy_model = energy_model

        self.base_optimizer = base_optimizer
        self.energy_optimizer = energy_optimizer

        self.base_scheduler = base_scheduler
        self.energy_scheduler = energy_scheduler

        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.custom_scheduler = custom_scheduler

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["base_model"] = self.base_model.state_dict()
        data["energy_model"] = self.energy_model.state_dict()

        if self.base_optimizer is not None:
            data["base_optimizer"] = self.base_optimizer.state_dict()
        if self.energy_optimizer is not None:
            data["energy_optimizer"] = self.energy_optimizer.state_dict()

        if self.base_scheduler is not None and not self.custom_scheduler:
            data["base_scheduler"] = self.base_scheduler.state_dict()
        if self.energy_scheduler is not None and not self.custom_scheduler:
            data["energy_scheduler"] = self.energy_scheduler.state_dict()

        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, with_optim=True, update_schedule=False, load_mapping={}, only_base=False):
        '''
        Parameters:
        ----------
            f : Filename to load from 
            with_optim: Boolean to indicate if the optimizer should be loaded as well
            update_schedule: 
            load_mapping: A dcitionary which specified mapping for some module to their names
            only_base: Indicate if only the base model should be loader
        '''
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_base_model(checkpoint, load_mapping, only_base)
        if not only_base:
            self._load_energy_model(checkpoint)
        if with_optim:
            if "base_optimizer" in checkpoint and self.base_optimizer:
                self.logger.info("Loading base_optimizer from {}".format(f))
                self.base_optimizer.load_state_dict(checkpoint.pop("base_optimizer"))
            if "energy_optimizer" in checkpoint and self.energy_optimizer:
                self.logger.info("Loading energy_optimizer from {}".format(f))
                self.energy_optimizer.load_state_dict(checkpoint.pop("energy_optimizer"))

            if "base_scheduler" in checkpoint and self.base_scheduler:
                self.logger.info("Loading base_scheduler from {}".format(f))
                if update_schedule:
                    self.base_scheduler.last_epoch = checkpoint["iteration"]
                else:
                    self.base_scheduler.load_state_dict(checkpoint.pop("base_scheduler"))
            if "energy_scheduler" in checkpoint and self.energy_scheduler:
                self.logger.info("Loading energy scheduler from {}".format(f))
                if update_schedule:
                    self.energy_scheduler.last_epoch = checkpoint["iteration"]
                else:
                    self.energy_scheduler.load_state_dict(checkpoint.pop("energy_scheduler"))
                

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_base_model(self, checkpoint, load_mapping, only_base):
        if only_base:
            key = "model"
        else:
            key = "base_model"
        load_state_dict(self.base_model, checkpoint.pop(key), load_mapping)

    def _load_energy_model(self, checkpoint):
        load_state_dict(self.energy_model, checkpoint.pop("energy_model"), {})
        # self.energy_model.load_state_dict(checkpoint.pop("energy_model"))
        