import datetime
import glob
import os

import torch

from modules.data.vocab import Vocab
from sys_config import TRAINED_PATH, EXP_DIR


def trim_hidden(inputs, hidden):
    """
    In the case where the last batch is smaller than batch_size,
    we will need to keep only the first N hidden states,
    where N is the number of samples in the last batch
    Args:
        inputs: the inputs in the last batch size
        hidden: the hidden state produced by the penultimate batch

    Returns: hidden
        the trimmed hidden state

    """
    batch_size = inputs.size(0)

    if isinstance(hidden, tuple):
        hidden_size = hidden[0].size(1)
    else:
        hidden_size = hidden.size(1)

    # do nothing
    if batch_size == hidden_size:
        return hidden

    # trim the hidden state to the remaining samples in the batch
    if isinstance(hidden, tuple):
        hidden = (hidden[0][:, :batch_size, :].contiguous(),
                  hidden[1][:, :batch_size, :].contiguous())
    else:
        hidden = hidden[:, :batch_size, :].contiguous()

    return hidden


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(v.detach() for v in h)


def sort_by_lengths(lengths):
    """
    Sort batch data and labels by length.
    Useful for variable length inputs, for utilizing PackedSequences
    Args:
        lengths (nn.Tensor): tensor containing the lengths for the data

    Returns:
        - sorted lengths Tensor
        - sort (callable) which will sort a given iterable
            according to lengths
        - unsort (callable) which will revert a given iterable to its
            original order

    """
    batch_size = lengths.size(0)

    sorted_lengths, sorted_idx = lengths.sort()
    _, original_idx = sorted_idx.sort(0, descending=True)
    reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

    if lengths.data.is_cuda:
        reverse_idx = reverse_idx.cuda()

    sorted_lengths = sorted_lengths[reverse_idx]

    def sort(iterable):
        if len(iterable.shape) > 1:
            return iterable[sorted_idx.data][reverse_idx]
        else:
            return iterable

    def unsort(iterable):
        if len(iterable.shape) > 1:
            return iterable[reverse_idx][original_idx][reverse_idx]
        else:
            return iterable

    return sorted_lengths, sort, unsort


def save_checkpoint(state, name, path=None, timestamp=False, tag=None,
                    verbose=False):
    """
    Save a trained model, along with its optimizer, in order to be able to
    resume training
    Args:
        path (str): the directory, in which to save the checkpoints
        timestamp (bool): whether to keep only one model (latest), or keep every
            checkpoint

    Returns:

    """
    now = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    if tag is not None:
        if isinstance(tag, str):
            name += "_{}".format(tag)
        elif isinstance(tag, list):
            for t in tag:
                name += "_{}".format(t)
        else:
            raise ValueError("invalid tag type!")

    if timestamp:
        name += "_{}".format(now)

    name += ".pt"

    if path is None:
        path = TRAINED_PATH

    file = os.path.join(path, name)

    if verbose:
        print("saving checkpoint:{} ...".format(name))

    torch.save(state, file)

    return file


def load_checkpoint(name, path=None):
    """
    Load a trained model, along with its optimizer
    Args:
        name (str): the name of the model
        path (str): the directory, in which the model is saved

    Returns:
        model, optimizer

    """
    if path is None:
        path = TRAINED_PATH

    # model_fname = os.path.join(path, "{}.pt".format(name))
    try:
        model_fname = os.path.join(path, name)
        print("Loading checkpoint `{}` ...".format(model_fname), end=" ")
        with open(model_fname, 'rb') as f:
            state = torch.load(f, map_location="cpu")
        print("done!")
    except:
        print("Loading checkpoint `{}` ...".format(name), end=" ")
        with open(name, 'rb') as f:
            state = torch.load(f, map_location="cpu")
        print("done!")

    if "vocab" in state:
        if isinstance(state["vocab"], tuple):
            state["vocab"] = tuple(Vocab().from_vocab_instance(v)
                                   for v in state["vocab"])
        else:
            state["vocab"] = Vocab().from_vocab_instance(state["vocab"])

    return state


def load_state_by_id(name, id=None):
    if id is None:
        return None

    exp_dir = os.path.join(EXP_DIR, name)
    states = [s for s in glob.glob(exp_dir + "/*" + id) if s.endswith(id)]

    if len(states) == 1:
        try:
            model_fname = glob.glob(states[0] + f"/{name}*best.pt")[0]
            print("Loading checkpoint `{}` ...".format(model_fname), end=" ")

            with open(model_fname, 'rb') as f:
                state = torch.load(f, map_location="cpu")

            print("done!")

            return state
        except:
            print(f"Found {states[0]}, but no checkpoint!")
            return None
    else:
        return None
