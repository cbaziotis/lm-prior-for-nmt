import html
import os

import numpy


def viz_row(sample):
    for item in sample:
        print(item)


def scores2opacities(opacities, length):
    if opacities is None:
        opacities = ["0"] * length
    else:
        opacities = numpy.array(opacities)
        # opacities = opacities / sum(numpy.exp(opacities))
        opacities = (opacities - opacities.mean()) / (4 * opacities.std())
        # opacities = numpy.exp(opacities)/sum(numpy.exp(opacities))

    return opacities


def sample2td(tokens, color=None, opacities=None, normalize=True):
    if normalize:
        opacities = scores2opacities(opacities, len(tokens))

    text = ""
    for t, o in zip(tokens, opacities):
        text += f"""
        <td style='background-color: rgba({color}, {o})'>{html.escape(t)}</td>
        """
    return text


def sample2span(tokens, color=None, opacities=None, normalize=True):
    if normalize:
        opacities = scores2opacities(opacities, len(tokens))

    text = ""
    for t, o in zip(tokens, opacities):
        text += f"""
        <span class='word' 
        style='background-color: rgba({color}, {o})'>{html.escape(t)}</span>
        """
    return text


def viz_samples(outputs, aligned):
    txt = ""

    if aligned:
        table = ""
        for out in outputs:
            row = f"<td>{out['tag']} ({len(out['tokens'])}):</td>"

            if 'opacity' not in out:
                out['opacity'] = None
            if 'color' not in out:
                out['color'] = None
            if 'normalize' not in out:
                out['normalize'] = True

            row += sample2td(out['tokens'], out['color'], out['opacity'],
                             out['normalize'])
            table += f"<tr>{row}</tr>"

        txt += f"<table>{table}</table>"
        return f"<div class='sample' style='display:block;'>{txt}</div>"
    else:

        for out in outputs:
            row = f"{out['tag']} ({len(out['tokens'])}):"

            if 'opacity' not in out:
                out['opacity'] = None
            if 'color' not in out:
                out['color'] = None
            if 'normalize' not in out:
                out['normalize'] = True

            row += sample2span(out['tokens'], out['color'], out['opacity'],
                             out['normalize'])
            txt += f"<div class='sentence'>{row}</div>"

        return f"<div class='sample' style='display:inline-block;'>{txt}</div>"


def samples2html(samples, aligned=False):
    dom = """
        <style>
        body {
          font-family: "CMU Serif", serif;
          font-size: 14px;
        }
        
        .samples-container {
          background: white;
          background-color: white;
          font-size: 12px;
          color: black;
        }
        
        .word {
          padding: 2px;
          margin: 0px;
          display: inline-block;
          color: black;
        }
        
        td {
          text-align: center;
          padding: 2px;
          font-size: 12px;
          white-space: nowrap;
        }
        
        .sentence {
          padding: 2px 0;
          float: left;
          width: 100%;
          display: inline-block;
        }
        
        .sample {
          border: 1px solid grey;
          padding: 0px 4px;
        }

        </style>
        <div class='samples-container'>
        """

    for s in samples:
        dom += viz_samples(s, aligned)

    dom += """
        </div>
        """
    return dom


def viz_html(dom):
    # or simply save in an html file and open in browser
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'attention.html')

    with open(file, 'w') as f:
        f.write(dom)
