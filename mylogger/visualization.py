import numpy
from visdom import Visdom


class Visualizer:

    def __init__(self, env="main",
                 server="http://localhost",
                 port=8097,
                 base_url="/",
                 http_proxy_host=None,
                 http_proxy_port=None,
                 log_to_filename=None):
        self._viz = Visdom(env=env,
                           server=server,
                           port=port,
                           http_proxy_host=http_proxy_host,
                           http_proxy_port=http_proxy_port,
                           log_to_filename=log_to_filename,
                           use_incoming_socket=False)
        self._viz.close(env=env)
        self.plots = {}

    def plot_line(self, name, tag, title, value, step=None):
        if name not in self.plots:

            y = numpy.array([value, value])
            if step is not None:
                x = numpy.array([step, step])
            else:
                x = numpy.array([1, 1])

            opts = dict(
                title=title,
                xlabel='steps',
                ylabel=name
            )

            if tag is not None:
                opts["legend"] = [tag]

            self.plots[name] = self._viz.line(X=x, Y=y, opts=opts)
        else:

            y = numpy.array([value])
            x = numpy.array([step])

            self._viz.line(X=x, Y=y, win=self.plots[name], name=tag,
                           update='append')

    def plot_text(self, text, title, pre=True):
        _width = max([len(x) for x in text.split("\n")]) * 10
        _heigth = len(text.split("\n")) * 20
        _heigth = max(_heigth, 120)
        if pre:
            text = "<pre>{}</pre>".format(text)

        self._viz.text(text, win=title, opts=dict(title=title,
                                                  width=min(_width, 450),
                                                  height=min(_heigth, 300)))

    def plot_bar(self, data, labels, title):
        self._viz.bar(win=title, X=data,
                      opts=dict(legend=labels, stacked=False, title=title))

    def plot_hist(self, data, title, numbins=20):
        self._viz.histogram(win=title, X=data,
                            opts=dict(numbins=numbins,
                                      title=title))

    def plot_scatter(self, data, title, targets=None, labels=None):
        self._viz.scatter(win=title, X=data, Y=targets,
                          opts=dict(
                              # legend=labels,
                              title=title,
                              markersize=5,
                              webgl=True,
                              width=400,
                              height=400,
                              markeropacity=0.5))

    def plot_heatmap(self, data, labels, title):

        height = min(data.shape[0] * 20, 600)
        width = min(data.shape[1] * 25, 600)

        self._viz.heatmap(win=title,
                          X=data,
                          opts=dict(
                              # title=title,
                              columnnames=labels[1],
                              rownames=labels[0],
                              width=width,
                              height=height,
                              layoutopts={'plotly': {
                                  'showscale': False,
                                  'showticksuffix': False,
                                  'showtickprefix': False,
                                  'xaxis': {
                                      'side': 'top',
                                      'tickangle': -60,
                                      # 'autorange': "reversed"
                                  },
                                  'yaxis': {
                                      'autorange': "reversed"
                                  },
                              }
                              }
                          ))
