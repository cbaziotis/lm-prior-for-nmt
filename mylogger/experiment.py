import glob
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy

from mylogger.helpers import dict_to_html, files_to_dict
from mylogger.visualization import Visualizer
from sys_config import VIS, EXP_DIR


class Experiment(object):
    """
    Experiment class
    """

    def __init__(self, name,
                 config,
                 desc=None,
                 src_dirs=None,
                 resume_state_id=None,
                 **kwargs):
        """

        Metrics = history of values
        Values = state of values
        Args:
            name:
            config:
            desc:
            output_dir:
            src_dirs:
        """
        self.name = name
        self.desc = desc
        self.config = config
        self.metrics = {}
        self.values = {}
        self.finished = False

        # the src files (dirs) to backup
        if src_dirs is not None:
            self.src = files_to_dict(src_dirs)
        else:
            self.src = None

        # the currently running script
        self.src_main = sys.argv[0]

        self.timestamp_start = datetime.now()
        self.timestamp_update = datetime.now()
        self.last_update = time.time()

        server = VIS["server"]
        port = VIS["port"]
        base_url = VIS["base_url"]
        http_proxy_host = VIS["http_proxy_host"]
        http_proxy_port = VIS["http_proxy_port"]
        self.enabled = VIS["enabled"]

        # --------------------------------------------------------------
        # create experiment instance directory
        # --------------------------------------------------------------
        exp_dir = os.path.join(EXP_DIR, self.name)

        if resume_state_id is None:
            self.output_dir = os.path.join(exp_dir, self.get_timestamp())
        else:
            states = glob.glob(exp_dir + "/*" + resume_state_id)
            if len(states) > 0:
                self.output_dir = states[0]
            else:
                self.output_dir = os.path.join(exp_dir, self.get_timestamp())
                self.output_dir += "_" + resume_state_id

        os.makedirs(self.output_dir, exist_ok=True)

        # --------------------------------------------------------------

        vis_log_file = os.path.join(self.output_dir, f"{self.name}.vis")

        if self.enabled:
            vis_id = os.path.normpath(self.output_dir).split(os.sep)[-2:]
            self.viz = Visualizer(env="_".join(vis_id),
                                  server=server,
                                  port=port,
                                  base_url=base_url,
                                  http_proxy_host=http_proxy_host,
                                  http_proxy_port=http_proxy_port,
                                  log_to_filename=vis_log_file)

            self.values["config"] = dict_to_html(self.config)
            self.text("config", self.values["config"], "Config")

    def save_line_fs(self, metric):
        fig, ax = plt.subplots()

        data = self.metrics[metric]
        if isinstance(data, dict):
            for k, v in data.items():
                ax.plot(numpy.arange(1, len(v) + 1, 1), v, label=k)
            plt.legend()
            ax.set_ylabel(metric)
        else:
            ax.plot(numpy.arange(1, len(self.metrics[metric]) + 1, 1),
                    self.metrics[metric])
            ax.set_xlabel('steps')
            ax.set_ylabel(metric)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{metric}.svg"),
                    bbox_inches='tight', format="svg")
        plt.close(fig)

    def save_hist_fs(self, key, data, title, numbins):
        fig, ax = plt.subplots()

        plt.hist(data, bins=numbins)
        plt.title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"{key}.svg"),
                    bbox_inches='tight', format="svg")
        plt.close(fig)

    def save_text_fs(self, key, text, title, pre):

        if re.match(".*\\<[^>]+>.*", text):
            extention = "html"
        else:
            extention = "txt"

        _path = os.path.join(self.output_dir, f"{key}.{extention}")
        with open(_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def line(self, name, tag, title, value, step=None):

        if name not in self.metrics:

            if tag is not None:
                self.metrics[name] = defaultdict(list)
                self.metrics[name][tag].append(value)
            else:
                self.metrics[name] = []
                self.metrics[name].append(value)

            if self.enabled:
                self.viz.plot_line(name, tag, title, value)

        else:

            if tag is not None:
                self.metrics[name][tag].append(value)
                size = len(self.metrics[name][tag])
            else:
                self.metrics[name].append(value)
                size = len(self.metrics[name])

            if step is None:
                step = size

            if self.enabled:
                self.viz.plot_line(name, tag, title, value, step)

        self.save_line_fs(name)

    def heatmap(self, key, data, labels, title):

        self.values[key] = (data, labels, title)

        if self.enabled:
            self.viz.plot_heatmap(data, labels, title)

    def histogram(self, key, data, title, numbins=20):

        self.values[key] = (data, title)

        self.save_hist_fs(key, data, title, numbins)

        if self.enabled:
            self.viz.plot_hist(data, title, numbins)

    def scatter(self, key, data, title, targets=None, labels=None):

        self.values[key] = (data, targets, labels, title)

        # self.save_scatter_fs(key, data, title, labels=None)

        if self.enabled:
            self.viz.plot_scatter(data, title, targets, labels)

    def text(self, key, text, title, pre=True):

        if not self.enabled:
            return

        self.values[key] = (text, title)
        self.viz.plot_text(text, title, pre)
        self.save_text_fs(key, text, title, pre)

    #############################################################
    # Persistence
    #############################################################
    def to_json(self):
        filename = os.path.join(self.output_dir, self.name + ".json")
        _state = {"metrics": self.metrics,
                  "name": self.name,
                  "finished": self.finished,
                  "config": self.config}

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(_state))

    def get_timestamp(self):
        return self.timestamp_start.strftime("%y-%m-%d_%H:%M:%S")

    def to_pickle(self):
        filename = os.path.join(self.output_dir, self.name + ".pickle")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def has_finished(self):
        try:
            filename = os.path.join(self.output_dir, self.name + ".json")
            with open(filename) as f:
                return json.load(f)["finished"]
        except:
            return False

    def finalize(self):
        self.finished = True
        self.save()

    def save(self):
        try:
            self.to_pickle()
        except:
            print("Failed to save to pickle...")

        try:
            self.to_json()
        except Exception as e:
            print("Failed to save to json...", e)
