import math
from collections import OrderedDict

import pandas
import torch
from numpy import mean
from tabulate import tabulate
from torch.nn import functional as F

from modules.data.loaders import MultiDataLoader
from modules.helpers import param_grad_wrt_loss


class TrainerCallback(object):
    def __init__(self, interval, **kwargs):
        self.interval = interval

    @staticmethod
    def get_step(trainer, losses):

        if isinstance(trainer.train_loader, MultiDataLoader):
            name = trainer.train_loader.get_current_loader()
            step = len([x for x in losses if x["loader"] == name])
            return step
        else:
            return trainer.step

    def skip_batch(self, trainer, losses):
        step = self.get_step(trainer, losses)
        skip = step % self.interval != 0
        return skip

    def batch_forward_end(self, *args):
        pass

    def batch_backward_end(self, *args):
        pass

    def batch_end(self, *args):
        pass

    def train_epoch_end(self, *args):
        pass

    def eval_epoch_end(self, *args):
        pass


class FunctionCallback(TrainerCallback):
    def __init__(self, func, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.func = func

    def batch_end(self, t, batch, epoch_losses, batch_losses, batch_outputs):
        # skip
        if self.get_step(t, epoch_losses) % self.interval != 0:
            return

        with torch.no_grad():
            try:
                self.func(t)
            except Exception as e:
                print(e)


# Print table with gradient norms wrt each weigh
def color_zero_red(val):
    color = 'red' if val == 0.0 else 'black'
    return 'color: %s' % color


class GradientCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_backward_end(self, t, batch, epoch_losses, batch_losses,
                           batch_outputs):
        # skip
        if self.get_step(t, epoch_losses) % self.interval != 0:
            return

        with torch.no_grad():
            df = pandas.DataFrame(t.grads(), columns=['Parameters', '∥∇∥'])
        df = df.sort_values(by=['∥∇∥'], ascending=False)
        table = (df.style.applymap(color_zero_red, subset=['∥∇∥'])
                 .set_properties(**{'text-align': 'left'})
                 .format("{:.4f}", subset=['∥∇∥'])
                 .bar(subset=['∥∇∥'], color='#d65f5f')
                 .hide_index()
                 .render())

        _key = f"grad_norms"
        _title = f"Gradient Norms"

        if isinstance(t.train_loader, MultiDataLoader):
            tag = t.train_loader.get_current_loader()
            _key += "_" + tag
            _title += ": " + tag

        t.exp.text(_key, table, _title)


class StatusCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch_end(self, trainer, losses):
        if trainer.failed_batches > 0:
            print()
            trainer.failed_batches = 0

        trainer.exp.text(f"status", f"Failed batches:{trainer.failed_batches}",
                         f"status")


class LossWeightCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_forward_end(self, t, batch, epoch_losses, batch_losses,
                          batch_outputs):
        # skip
        if self.get_step(t, epoch_losses) % self.interval != 0:
            return

        for k, w in t.loss_weights.items():
            t.exp.line(f"weight_values", k,
                       f"Loss Weight Values", t.anneal_step(w))


class ModuleGradientCallback(TrainerCallback):
    def __init__(self, modules, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.modules = modules

        if self.modules is None:
            self.modules = []

    def batch_forward_end(self, t, batch, epoch_losses, batch_losses,
                          batch_outputs):
        # skip
        if self.get_step(t, epoch_losses) % self.interval != 0:
            return

        with torch.no_grad():
            norms = {tag: param_grad_wrt_loss(t.optimizers, t.model, loss)
                     for tag, loss in batch_losses.items()}
        df = pandas.DataFrame(norms)
        table = (df.style.applymap(color_zero_red)
                 .set_properties(**{'text-align': 'left'})
                 .format("{:.4f}")
                 # .bar(color='#d65f5f', axis=None)
                 .bar(color='#d65f5f', axis=0)
                 .render())

        _key = f"grads_per_loss"
        _title = f"Gradients Per Loss"

        if isinstance(t.train_loader, MultiDataLoader):
            tag = t.train_loader.get_current_loader()
            _key += "_" + tag
            _title += ": " + tag

        t.exp.text(_key, table, _title)

        for module in self.modules:
            for loss, props in t.config["losses"].items():
                module_norms = [norm for param, norm in norms[loss].items()
                                if module in param]

                tags = []
                if len(self.modules) > 1:
                    tags.append(module)

                if len(t.config["losses"]) > 1:
                    tags.append(loss)

                if len(tags) > 0:
                    tag = "-".join(tags)
                else:
                    tag = None

                t.exp.line(f"{module}_grad_norms", tag,
                           f"Module '{module}' ∥∇∥", mean(module_norms))


class EmbeddingInspectionCallback(TrainerCallback):
    def __init__(self, embeddings, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.embeddings = embeddings

    def batch_end(self, t, batch, epoch_losses, batch_losses, batch_outputs):
        # skip
        if (t.step % self.interval != 0) and t.step != 1:
            return

        with torch.no_grad():
            centroid = torch.mean(self.embeddings, axis=0)
            angles = F.cosine_similarity(self.embeddings, centroid.unsqueeze(0))
            # fro_norm = torch.norm(self.embeddings, p='fro')
            centroid_norm = torch.norm(centroid)
            emb_norms = torch.norm(self.embeddings, dim=1)
            del centroid

        t.exp.line("centroid_norm", None, "Norm of Centroid",
                   centroid_norm.item())
        # t.exp.line("matrix_norm", None, "Frobenius Norm of Embeddings",
        #               fro_norm.item())
        t.exp.histogram("angles", angles.data.cpu().numpy(),
                        "Distribution of Cosines with Centroid", 100)
        t.exp.histogram("emb_norms", emb_norms.data.cpu().numpy(),
                        "Distribution of Norms", 100)

        # del angles, fro_norm, centroid_norm, emb_norms
        del angles, emb_norms, centroid_norm


class EmbeddingNormCallback(TrainerCallback):
    def __init__(self, embeddings, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.weights = embeddings

    def batch_end(self, t, batch, epoch_losses, batch_losses, batch_outputs):
        # skip
        if (t.step % self.interval != 0) and t.step != 1:
            return

        with torch.no_grad():
            self.weights.data -= self.weights.data.mean(0)


class EmbeddingMaxNormCallback(TrainerCallback):
    def __init__(self, embeddings, max_norm, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.weights = embeddings
        self.max_norm = max_norm

    def batch_end(self, t, batch, epoch_losses, batch_losses, batch_outputs):
        # skip
        if (t.step % self.interval != 0) and t.step != 1:
            return

        with torch.no_grad():
            norms = torch.norm(self.weights.data, 2, dim=1)
            _max = torch.ones_like(norms) * self.max_norm
            self.weights.data.div_(torch.max(norms, _max).unsqueeze(1))


class CheckpointCallback(TrainerCallback):
    def __init__(self, interval,
                 only_best=False,
                 threshold=None,
                 early_stop=100,
                 **kwargs):
        super().__init__(interval, **kwargs)
        self.only_best = only_best
        self.best = threshold
        self.early_stop = early_stop
        self.patience = early_stop

    def eval_epoch_end(self, trainer, losses):
        val_loss = pandas.DataFrame(losses).mean().mean()
        is_best = False

        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.patience = self.early_stop
            is_best = True

        else:
            self.patience -= 1

            if self.patience < 0:
                trainer.early_stop = True

        if is_best:
            trainer.checkpoint(name=trainer.config["name"], tags=["best"])
            # trainer.exp.values['epoch_progress']

        elif not self.only_best:
            trainer.checkpoint(name=trainer.config["name"],
                               tags=[trainer.epoch, trainer.step])


class LossCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_log_valid = ""
        self.last_log_train = ""
        self.best = None
        self.omit_steps = 2

    @staticmethod
    def _apply_prefix(key, prefix, sep):

        if len(prefix) > 0:
            return sep.join([prefix, key])
        else:
            return key

    def _run(self, t, losses, loader, loss_prefix="", tag_prefix=""):

        n_losses = len(t.config["losses"])
        log = []  # list of records with each loss

        # ------------------------------------------------------------------
        # Multiple Datasets
        # ------------------------------------------------------------------
        if isinstance(loader, MultiDataLoader):
            losses = pandas.DataFrame(losses).groupby("loader").mean().to_dict()

            # for each loss
            for loss_name, loss_values in losses.items():

                # for each data_loader
                for loader_name, loss in loss_values.items():

                    if math.isnan(loss):
                        continue

                    _tag = loss_name.upper() + " - " + loader_name.upper()
                    if len(tag_prefix) > 0:
                        _tag = tag_prefix.upper() + " - " + _tag

                    _key = self._apply_prefix("loss", loss_prefix.lower(), "_")
                    _title = self._apply_prefix("Loss", loss_prefix, " ")

                    # print line plot with loss
                    # global
                    t.exp.line(_key, _tag, _title, loss)
                    # loader-specific
                    t.exp.line(f"{_key}_{loader_name}", loss_name.upper(),
                               f"{_title} {loader_name}", loss)

                    record = OrderedDict({"Tag": _tag, "Loss": loss})

                    # for sequence-level tasks
                    if t.config["losses"][loss_name].get("perplexity", False):
                        ppl = math.exp(loss)
                        _key = self._apply_prefix("ppl", loss_prefix.lower(),
                                                  "_")
                        _title = self._apply_prefix("Perplexity", loss_prefix,
                                                    " ")
                        # print line plot with perplexity
                        # global
                        t.exp.line(_key, _tag, _title, ppl)
                        # loader-specific
                        t.exp.line(f"{_key}_{loader_name}",
                                   loss_name.upper(),
                                   f"{_title} {loader_name}", ppl)
                        record["Perplexity"] = ppl

                    log.append(record)

        # ------------------------------------------------------------------
        # Single Dataset
        # ------------------------------------------------------------------
        else:
            losses = pandas.DataFrame(losses).mean().to_dict()
            for name, loss in losses.items():

                if math.isnan(loss):
                    continue

                record = OrderedDict()

                _tag = None

                if n_losses > 1:
                    _tag = name.upper()

                if len(tag_prefix) > 0:
                    if n_losses > 1:
                        _tag = tag_prefix.upper() + " - " + _tag
                    else:
                        _tag = tag_prefix.upper()

                if _tag is not None:
                    record["Tag"] = _tag

                _key = self._apply_prefix("loss", loss_prefix.lower(), "_")
                _title = self._apply_prefix("Loss", loss_prefix, " ")

                # print line plot with loss
                t.exp.line(_key, _tag, _title, loss)
                record["Loss"] = loss

                # for sequence-level tasks
                if t.config["losses"][name].get("perplexity", False):
                    ppl = math.exp(loss)

                    _key = self._apply_prefix("ppl", loss_prefix.lower(), "_")
                    _title = self._apply_prefix("Perplexity", loss_prefix, " ")

                    # print line plot with perplexity
                    t.exp.line(_key, _tag, _title, ppl)
                    record["Perplexity"] = ppl

                log.append(record)

        # ------------------------------------------------------------------
        # Log results in table format
        # ------------------------------------------------------------------
        if len(log) == 1:
            log = "\t".join(["{:}: {:8.2f}".format(k, v) if isinstance(v, float)
                             else v
                             for k, v in log[0].items()])
        else:
            log = tabulate(log,
                           headers="keys", floatfmt=".4f", numalign="right")

        return log

    def _update_epoch_log(self, t):
        t.exp.text("epoch_progress",
                   "\n".join([self.last_log_train, self.last_log_valid]),
                   "Epoch Progress", pre=True)

    def train_epoch_end(self, trainer, losses):
        log = self._run(trainer, losses, trainer.train_loader, "Epoch", "TRAIN")
        print()
        header = f"Epoch {trainer.epoch}: Average training statistics" + "\n"
        header += "-" * 40 + "\n"
        log = header + log + "\n"
        print(log)
        self.last_log_train = log
        self._update_epoch_log(trainer)

    def eval_epoch_end(self, trainer, losses):
        log = self._run(trainer, losses, trainer.valid_loader, "Epoch", "VALID")
        print()
        header = f"Epoch {trainer.epoch}: Average validation statistics" + "\n"
        header += "-" * 40 + "\n"
        log = header + log + "\n"
        print(log)
        self.last_log_valid = log
        self._update_epoch_log(trainer)

        _mean_loss = trainer.aggregate_eval_losses(losses)
        if self.best is None or _mean_loss < self.best:
            self.best = _mean_loss

            trainer.exp.text("best_epoch", "\n".join([self.last_log_train,
                                                      self.last_log_valid]),
                             "Best Epoch", pre=True)

    def batch_backward_end(self, t, batch, epoch_losses, batch_losses,
                           batch_outputs):

        # skip
        if t.step % self.interval != 0:
            return

        if self.omit_steps > 0:
            self.omit_steps -= 1
            return

        log = self._run(t, epoch_losses[-self.interval:], t.train_loader)

        t.exp.text("progress", t.progress_log + "\n" + log, "Progress")

        # clean lines and move cursor back up N lines
        print("\n\033[K" + log)
        print("\033[F" * (len(log.split("\n")) + 6))
