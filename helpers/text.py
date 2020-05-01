from itertools import groupby


def devectorize(data, id2tok, eos=None, strip_eos=False, oov_map=None, pp=False):
    if strip_eos:
        for i in range(len(data)):
            try:
                data[i] = data[i][:list(data[i]).index(eos)]
            except:
                continue

    # ids to words
    data = [[id2tok.get(x, "<unk>") for x in seq] for seq in data]

    if oov_map is not None:
        data = [[m.get(x, x) for x in seq] for seq, m in zip(data, oov_map)]

    if pp:
        rules = {f"<oov-{i}>": "UNK" for i in range(10)}
        rules["unk"] = "UNK"
        rules["<unk>"] = "UNK"
        rules["<sos>"] = ""
        rules["<eos>"] = ""
        rules["<pad>"] = ""

        data = [[rules.get(x, x) for x in seq] for seq in data]

        # remove repetitions
        data = [[x[0] for x in groupby(seq)] for seq in data]

    return data


def detokenize(x, subword):
    if subword:
        return ''.join(x).replace('▁', ' ').lstrip().rstrip()
    else:
        return " ".join(x)
