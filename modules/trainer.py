import gc
import time
from typing import List

import numpy
import pandas
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, \
    StepLR, MultiStepLR

from helpers._logging import epoch_progress
from helpers.generic import number_h
from helpers.training import save_checkpoint, load_state_by_id
from libs.joeynmt.builders import NoamScheduler
from modules.data.loaders import MultiDataLoader
from modules.optim.lookahead import Lookahead
from modules.optim.radam import RAdam
from modules.callbacks import TrainerCallback
from mylogger.experiment import Experiment


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 valid_loader,
                 config,
                 device,
                 callbacks: List[TrainerCallback] = None,
                 resume_state_id: str = None,
                 resume_state=None,
                 **kwargs):

        self.config = config
        self.device = device
        self.epoch = 0
        self.step = 0
        self.failed_batches = 0
        self.early_stop = False
        self.progress_log = None
        self.best_checkpoint = None
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_batches = len(train_loader)
        self.total_steps = self.n_batches * self.config["epochs"]
        self.model = model
        self.callbacks = callbacks

        # -----------------------------------------------------------------
        # Optimization
        # -----------------------------------------------------------------
        self.optimizers = self.__init_optimizer(self.config["optim"])
        if len(self.optimizers) == 1:
            self.scheduler = self.__init_scheduler(self.config["optim"])

            if self.config["optim"]["scheduler"] == "noam":
                self.scheduler.step()

        else:
            self.scheduler = None
            raise print("Currently schedulers support only 1 optimizer!")

        self.loss_weights = self.__init_loss_weights()

        # -----------------------------------------------------------------
        # Model definition
        # -----------------------------------------------------------------
        self.model.to(device)
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters()
                                     if p.requires_grad)

        print("Total Params:", number_h(total_params))
        print("Total Trainable Params:", number_h(total_trainable_params))

        # -----------------------------------------------------------------
        # Experiment definition - Resume training from interrupted state
        # -----------------------------------------------------------------
        if resume_state_id is not None:
            resume_state = load_state_by_id(self.config["name"],
                                            resume_state_id)

        self.model_type = self.config["model"].get("type", "rnn")

        if resume_state is not None:
            self.exp = resume_state["exp"]
            self.load_state(resume_state)

            if self.exp.has_finished():
                print("Experiment is already finished!")

            try:
                model.tie_weights()
            except:
                pass
            print(f"Resuming from previous state with id:{resume_state_id}...")

        else:
            print(f"Starting with state id:{resume_state_id}...")
            self.exp = Experiment(self.config["name"], config,
                                  src_dirs=kwargs.get("src_dirs"),
                                  resume_state_id=resume_state_id)

        # print initial learning rate
        self.exp.line("lr", None, "Learning Rate",
                      self.optimizers[0].param_groups[0]['lr'])

    def _cyclical_schedule(self, cycle, n_cycles, floor=0., ceil=1., start=0):
        warm = [floor] * start
        anneal = numpy.linspace(floor, ceil, cycle).tolist() * n_cycles
        anneal = anneal[:self.total_steps]
        end = [ceil] * (self.total_steps - len(anneal))
        return warm + anneal + end

    def _linear_schedule(self, start, stop, floor=0., ceil=1.):
        warm = [floor] * start
        anneal = numpy.linspace(floor, ceil, stop - start).tolist()
        end = [ceil] * (self.total_steps - stop)
        return warm + anneal + end

    def _sigmoid_schedule(self, start, stop, floor=0., ceil=1., k=10):
        warm = [ceil] * start
        anneal = numpy.linspace(ceil, floor, stop - start).tolist()
        anneal = [(k / (k + numpy.exp(i / k))) for i, x in enumerate(anneal)]
        end = [floor] * (self.total_steps - stop)
        return warm + anneal + end

    def __init_loss_weights(self):
        loss_weights = dict()
        for loss, p in self.config["losses"].items():
            if isinstance(p["weight"], list):

                floor = float(p["weight"][0])
                ceil = float(p["weight"][1])

                if p["annealing_schedule"] == "cyclical":
                    cycle = p["annealing_cycle"]
                    n_cycles = self.total_steps // cycle
                    start = p.get("annealing_start", 0)
                    schedule = self._cyclical_schedule(cycle, n_cycles,
                                                       floor=floor, ceil=ceil,
                                                       start=start)
                    loss_weights[loss] = schedule

                elif p["annealing_schedule"] == "linear":
                    start = p.get("annealing_start", 0)
                    stop = p.get("annealing_stop", self.total_steps)
                    stop = min(self.total_steps, stop)
                    schedule = self._linear_schedule(start, stop, floor, ceil)

                    loss_weights[loss] = schedule

                elif p["annealing_schedule"] == "sigmoid":
                    start = p.get("annealing_start", 0)
                    stop = p.get("annealing_stop", self.total_steps)
                    stop = min(self.total_steps, stop)
                    schedule = self._linear_schedule(start, stop, floor, ceil)

                    loss_weights[loss] = schedule
            else:
                loss_weights[loss] = p["weight"]

        return loss_weights

    def anneal_init(self, param):
        if isinstance(param, list):
            if len(param) == 2:
                steps = self.total_steps
            else:
                steps = param[2]
            return numpy.linspace(param[0], param[1], num=steps).tolist()
        else:
            return param

    def anneal_step(self, param):
        if isinstance(param, list):
            try:
                _val = param[self.step]
            except:
                _val = param[-1]
        else:
            _val = param

        return _val

    def __tensors_to_device(self, batch):
        return list(map(lambda x: x.to(self.device,
                                       non_blocking=False) if x is not None else x,
                        batch))

    def batch_to_device(self, batch):

        """
        Move batch tensors to model's device
        """
        if torch.is_tensor(batch[0]):
            batch = self.__tensors_to_device(batch)
        else:
            batch = list(map(lambda x: self.__tensors_to_device(x), batch))

        return batch

    def _aggregate_losses(self, batch_losses):
        """
        This function computes a weighted sum of the models losses

        Returns:
            loss_sum (int): the aggregation of the constituent losses
            loss_list (list, int): the constituent losses

        """

        if isinstance(batch_losses, (tuple, list)):
            if self.loss_weights is None:
                _ws = [1.0 for _ in batch_losses]
            else:
                _ws = [self.anneal_step(w) for w in self.loss_weights]

            total = sum(w * x for x, w in zip(batch_losses, _ws)) / len(
                batch_losses)
            # losses = [w * x.item() for x, w in zip(batch_losses, _ws)]
            losses = [x.item() for x, w in zip(batch_losses, _ws)]

        elif isinstance(batch_losses, dict):
            if self.loss_weights is None:
                _ws = {n: 1.0 for n, _ in batch_losses.items()}
            else:
                _ws = {k: self.anneal_step(w) for k, w
                       in self.loss_weights.items()}

            total = sum(v * _ws[k] for k, v in batch_losses.items()) / len(
                batch_losses)
            # losses = {n: x.item() * _ws[n] for n, x in batch_losses.items()}
            losses = {n: x.item() for n, x in batch_losses.items()}

        else:
            total = batch_losses
            losses = batch_losses.item()

        return total, losses

    def __init_optimizer(self, config):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(parameters, lr=config["lr"],
                                         # betas=(0.9, 0.98), eps=1e-9,
                                         weight_decay=config["weight_decay"])

        elif config["optimizer"] == "radam":
            optimizer = RAdam(parameters, lr=config["lr"])
        elif config["optimizer"] == "ranger":
            base_optim = RAdam(parameters, lr=config["lr"])
            optimizer = Lookahead(base_optim, k=config["k"])
        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=config["lr"],
                                        weight_decay=config["weight_decay"])
        else:
            raise ValueError

        if not isinstance(optimizer, (tuple, list)):
            optimizer = [optimizer]

        return optimizer

    def __init_scheduler(self, config):
        if config["scheduler"] == "plateau":
            return ReduceLROnPlateau(self.optimizers[0], 'min',
                                     patience=config["patience"],
                                     factor=config["gamma"],
                                     verbose=True,
                                     min_lr=config["min_lr"])

        elif config["scheduler"] == "cosine":
            return CosineAnnealingLR(self.optimizers[0],
                                     T_max=self.config["epochs"],
                                     eta_min=config["eta_min"])

        elif config["scheduler"] == "step":
            return StepLR(self.optimizers[0],
                          step_size=config["step_size"],
                          gamma=config["gamma"])

        elif config["scheduler"] == "multistep":
            return MultiStepLR(self.optimizers[0],
                               milestones=config["milestones"],
                               gamma=config["gamma"])

        elif config["scheduler"] == "noam":
            return NoamScheduler(self.model.ninp,
                                 self.optimizers[0],
                                 factor=config.get("factor", 1),
                                 warmup=config.get("warmup", 8000))

        else:
            return None

    def step_scheduler(self, loss=None):
        if self.scheduler is not None:
            if self.config["optim"]["scheduler"] == "plateau":
                if loss is not None:
                    self.scheduler.step(loss)
            else:
                self.scheduler.step()

            if self.step % self.config["logging"]["log_interval"] == 0:
                self.exp.line("lr", None, "Learning Rate",
                              self.optimizers[0].param_groups[0]['lr'])

    def process_batch(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def cross_entropy_loss(logits, labels, lengths=None, ignore_index=0):

        """
        Compute a sequence loss (i.e. per timestep).
        Used for tasks such as Translation, Language Modeling and
        Sequence Labelling.
        """

        _logits = logits.contiguous().view(-1, logits.size(-1))
        _labels = labels.contiguous().view(-1)

        if lengths is None:
            loss = F.cross_entropy(_logits, _labels, ignore_index=ignore_index)
            return loss

        else:
            _loss = F.cross_entropy(_logits, _labels, ignore_index=ignore_index,
                                    reduction='none')
            _loss_per_step = _loss.view(labels.size())
            loss = _loss.sum() / lengths.float().sum()
            return loss, _loss_per_step

    def grads(self):
        """
        Get the list of the norms of the gradients for each parameter
        """
        return [(name, parameter.grad.norm().item())
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None]

    @staticmethod
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def empty_batch_outputs(self, outputs):
        pass

    def train_step(self, batch, epoch_losses, batch_index, epoch_start):

        batch = self.batch_to_device(batch)

        # forward pass using the model-specific _process_batch()
        batch_losses, batch_outputs = self.process_batch(*batch)

        # ----------------------------------------------------------------
        # Callbacks: Batch Forward End
        # ----------------------------------------------------------------
        for c in self.callbacks:
            c.batch_forward_end(self, batch, epoch_losses,
                                batch_losses, batch_outputs)
        # ----------------------------------------------------------------

        # aggregate the losses
        loss_sum, loss_list = self._aggregate_losses(batch_losses)

        if isinstance(self.train_loader, MultiDataLoader):
            loss_list["loader"] = self.train_loader.get_current_loader()

        epoch_losses.append(loss_list)

        # back-propagate
        loss_sum.backward()

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if self.step % self.config["logging"]["log_interval"] == 0:
            self.progress_log = epoch_progress(self.epoch, batch_index,
                                               self.n_batches, epoch_start,
                                               self.config["name"])

        # Callbacks: Batch Backward End
        for c in self.callbacks:
            c.batch_backward_end(self, batch, epoch_losses, batch_losses,
                                 batch_outputs)
        # ----------------------------------------------------------------

        if self.config["optim"]["clip"] is not None:
            # clip_grad_norm_(self.model.parameters(), self.clip)
            for optimizer in self.optimizers:
                clip_grad_norm_((p for group in optimizer.param_groups
                                 for p in group['params']),
                                self.config["optim"]["clip"])

        # update weights
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad()

        # Callbacks: Batch Backward End
        for c in self.callbacks:
            c.batch_end(self, batch, epoch_losses, batch_losses,
                        batch_outputs)

        # ----------------------------------------------------------------
        # Explicitly free GPU memory
        # ----------------------------------------------------------------
        if batch_outputs is not None:
            self.empty_batch_outputs(batch_outputs)
            batch_outputs.clear()
        batch_losses.clear()
        del batch[:]
        del batch_losses, batch_outputs, batch, loss_sum

    def free_gpu(self):
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Failed to free GPU memory!")

    def train_epoch(self):
        """
        Train the network for one epoch and return the average loss.
        * This will be a pessimistic approximation of the true loss
        of the network, as the loss of the first batches will be higher
        than the true.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.train()
        epoch_losses = []
        self.epoch += 1
        epoch_start = time.time()
        # self.free_gpu()

        try:
            self.train_loader.reset()
        except:
            pass

        for batch_index, batch in enumerate(self.train_loader, 1):
            self.model.train()
            self.step += 1

            try:
                self.train_step(batch, epoch_losses, batch_index, epoch_start)
            except RuntimeError as e:
                print(f"Error processing batch: {batch_index}. Trying again...")
                print(e)
                try:
                    self.free_gpu()
                    self.train_step(batch, epoch_losses, batch_index,
                                    epoch_start)
                except RuntimeError as e:
                    self.failed_batches += 1
                    # print('| WARNING: failed again! skipping batch')
                    continue

            for c in self.callbacks:
                try:
                    c.batch_start(self)
                except Exception as e:
                    pass

            if self.config["optim"].get("interval", "epoch") == "batch":
                self.step_scheduler()

            if self.early_stop:
                break

            # explicitly free memory
            try:
                del batch[:]
            except:
                pass
            del batch

        for c in self.callbacks:
            c.train_epoch_end(self, epoch_losses)

        # self.free_gpu()
        self.exp.save()

        return epoch_losses

    def aggregate_eval_losses(self, losses):
        if "val_loss" in self.config["optim"]:
            _key = self.config["optim"]["val_loss"]
            return pandas.DataFrame(losses)[_key].mean()
        else:
            return pandas.DataFrame(losses).mean().sum()

    def eval_batch(self, batch):
        batch = self.batch_to_device(batch)
        batch_losses, batch_outputs = self.process_batch(*batch)
        # aggregate the losses into a single loss value
        loss, _losses = self._aggregate_losses(batch_losses)

        # -------------------------------------------
        # Explicitly free GPU memory
        # -------------------------------------------
        if batch_outputs is not None:
            self.empty_batch_outputs(batch_outputs)
            batch_outputs.clear()
        batch_losses.clear()
        del batch[:]
        del batch_losses, batch_outputs, batch, loss
        return _losses

    def eval_epoch(self, only_eval=False, custom_loader=None):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if custom_loader is not None:
            loader = custom_loader
        else:
            loader = self.valid_loader

        # todo: what is that????
        try:
            loader.reset()
        except:
            pass

        with torch.no_grad():
            for i_batch, batch in enumerate(loader, 1):

                try:
                    _losses = self.eval_batch(batch)
                except RuntimeError:
                    try:
                        self.free_gpu()
                        _losses = self.eval_batch(batch)
                    except RuntimeError as e:
                        raise e

                losses.append(_losses)

                # explicitly free memory
                try:
                    del batch[:]
                except:
                    pass
                del batch

        # just return the losses and skip the rest steps. useful for getting
        # the loss on the val set without waiting for the end of an epoch
        if only_eval:
            return losses

        for c in self.callbacks:
            c.eval_epoch_end(self, losses)

        if self.config["optim"].get("interval", "epoch") == "epoch":
            self.step_scheduler(self.aggregate_eval_losses(losses))

        return losses

    def get_vocab(self):
        raise NotImplementedError

    def get_state(self):
        """
        Return a dictionary with the current state of the model.
        The state should contain all the important properties which will
        be save when taking a model checkpoint.
        Returns:
            state (dict)

        """
        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "early_stop": self.early_stop,
            "callbacks": self.callbacks,
            "progress_log": self.progress_log,
            "best_checkpoint": self.best_checkpoint,
            "loss_weights": self.loss_weights,
            "exp": self.exp,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": self.get_vocab(),
        }

        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        else:
            state["scheduler"] = None

        return state

    def load_state(self, state):
        self.config = state["config"]
        self.epoch = state["epoch"]
        self.step = state["step"]
        self.early_stop = state["early_stop"]
        self.callbacks = state["callbacks"]
        self.progress_log = state["progress_log"]
        self.best_checkpoint = state["best_checkpoint"]
        self.loss_weights = state["loss_weights"]

        self.model.load_state_dict(state["model"])
        self.model.to(self.device)

        for i, opt in enumerate(self.optimizers):
            self.optimizers[i].load_state_dict(state["optimizers"][i])
            for s in self.optimizers[i].state.values():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        s[k] = v.to()

        if state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        else:
            self.scheduler = None

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        if self.exp is not None:
            self.exp.save()
            path = self.exp.output_dir
        else:
            path = None

        self.best_checkpoint = save_checkpoint(self.get_state(),
                                               path=path,
                                               name=name, tag=tags,
                                               timestamp=timestamp,
                                               verbose=verbose)
        return self.best_checkpoint
