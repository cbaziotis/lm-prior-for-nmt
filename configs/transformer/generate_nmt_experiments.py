def get_name(src, trg, model, ls=False, synthetic=False, subsample=0):
    name = f"final.trans.{src}{trg}_{model}"

    if ls:
        name += "_ls"

    if synthetic:
        name += "_synthetic"

    if subsample > 0:
        name += f"_{subsample}"

    return name


def _add_tags(pair, model_name, ls, bt, synth, subsample):
    src = pair[:2]
    trg = pair[2:]

    cmd = ""
    if subsample > 0:
        cmd += f" data.subsample={subsample}"

    if ls:
        cmd += " losses.mt.smoothing=0.1"

    if bt:
        cmd += " data.backtranslate_path="
        if src == "tr":
            cmd += f"../datasets/mono/priors/news.tr.pp.train"
        else:
            cmd += f"../datasets/mono/priors/news.{src}.2014-2017.pp.3M.train"

    if synth:
        cmd += " data.synthetic={}"

        back_model = get_name(trg, src, model_name, ls)
        cmd += " data.synthetic.src_path="
        cmd += f"../datasets/mono/priors/{back_model}.synthetic"

        cmd += " data.synthetic.trg_path="
        if trg == "tr":
            cmd += f"../datasets/mono/priors/news.tr.pp.train"
        else:
            cmd += f"../datasets/mono/priors/news.{trg}.2014-2017.pp.3M.train"

    name = get_name(src, trg, model_name, ls, synth, subsample)
    cmd += f" --name {name}"

    return cmd


def baseline(pair, ls, bt, synth, subsample=0):
    # specify base configuration file
    cmd = "python nmt_prior.py"
    cmd += f" --config ../../configs/acl_transformer/trans.{pair}_base.yaml"

    cmd += _add_tags(pair, "base", ls, bt, synth, subsample)

    return cmd


def fusion(pair, method, ls, bt, synth, subsample=0):
    src = pair[:2]
    trg = pair[2:]

    # specify base configuration file
    cmd = "python nmt_prior.py"
    cmd += f" --config ../../configs/acl_transformer/trans.{pair}_fusion.yaml"

    # specify which fusion method to use
    cmd += f" model.decoding.fusion={method}"

    # specify which prior to use
    cmd += f" data.prior_path="
    cmd += f"../../checkpoints/prior.lm_news_{trg}_3M_trans_big_best.pt"

    cmd += _add_tags(pair, method, ls, bt, synth, subsample)

    return cmd


def prior(pair, ls, bt, synth, subsample=0, obj="kl", size="30M"):
    src = pair[:2]
    trg = pair[2:]

    # specify base configuration file
    cmd = "python nmt_prior.py"
    cmd += f" --config ../../configs/acl_transformer/trans.{pair}_prior.yaml"

    # specify which regularization objective to use for the prior
    cmd += f" losses.prior.objective={obj}"

    # specify which prior to use
    cmd += f" data.prior_path="
    cmd += f"../../checkpoints/prior.lm_news_{trg}_{size}_trans_big_best.pt"

    cmd += _add_tags(pair, f"prior_{size}_{obj}", ls, bt, synth, subsample)

    return cmd


def lang_pair_experiments(p, bt, syn, subsample):
    exps = []

    if not syn and not bt:
        # baseline
        exps.append(
            baseline(p, ls=False, bt=bt, synth=syn, subsample=subsample))

    # baseline + LS
    exps.append(baseline(p, ls=True, bt=bt, synth=syn, subsample=subsample))

    if not syn and not bt:
        # LM-fusion
        exps.append(fusion(p, "prenorm", ls=False, bt=bt, synth=syn,
                           subsample=subsample))
        exps.append(fusion(p, "postnorm", ls=False, bt=bt, synth=syn,
                           subsample=subsample))

        # LM-fusion + LS
        exps.append(fusion(p, "prenorm", ls=True, bt=bt, synth=syn,
                           subsample=subsample))
        exps.append(fusion(p, "postnorm", ls=True, bt=bt, synth=syn,
                           subsample=subsample))

    # LM-Prior
    exps.append(prior(p, ls=False, bt=bt, synth=syn, obj="kl", size="3M",
                      subsample=subsample))
    # exps.append(prior(p, ls=False, bt=bt, synth=syn, obj="rkl", size="3M", subsample=subsample))

    # LM-Prior + LS
    exps.append(prior(p, ls=True, bt=bt, synth=syn, obj="kl", size="3M",
                      subsample=subsample))
    # exps.append(prior(p, ls=True, bt=bt, synth=syn, obj="rkl", size="3M", subsample=subsample))

    if p != "entr":
        # LM-Prior (30M)
        exps.append(prior(p, ls=False, bt=bt, synth=syn, obj="kl", size="30M",
                          subsample=subsample))
        # exps.append(prior(p, ls=False, bt=bt, synth=syn, obj="rkl", size="30M", subsample=subsample))

        # LM-Prior (30M) + LS
        exps.append(prior(p, ls=True, bt=bt, synth=syn, obj="kl", size="30M",
                          subsample=subsample))
        # exps.append(prior(p, ls=True, bt=bt, synth=syn, obj="rkl", size="30M", subsample=subsample))

    return exps


def generate_experiments(bt, syn):
    if bt:
        fname = "experiments_nmt_backtranslate.sh"
    elif syn:
        fname = "experiments_nmt_synthetic.sh"
    else:
        fname = "experiments_nmt.sh"

    with open(fname, "w") as f:
        for p in ["deen", "ende", "entr", "tren"]:
            for exp in lang_pair_experiments(p, bt, syn, subsample=0):
                f.write(exp + "\n")


def generate_subsample_experiments(p, bt, syn):
    if bt:
        fname = f"experiments_nmt_backtranslate_subsample_{p}.sh"
    elif syn:
        fname = f"experiments_nmt_synthetic_subsample_{p}.sh"
    else:
        fname = f"experiments_nmt_subsample_{p}.sh"

    with open(fname, "w") as f:
        for size in [10000, 50000, 100000]:
            for exp in lang_pair_experiments(p, bt, syn, subsample=size):
                f.write(exp + "\n")


if __name__ == "__main__":
    generate_experiments(bt=False, syn=False)
    generate_experiments(bt=True, syn=False)
    generate_experiments(bt=False, syn=True)

    generate_subsample_experiments("deen", bt=False, syn=False)
    generate_subsample_experiments("deen", bt=True, syn=False)
    generate_subsample_experiments("deen", bt=False, syn=True)
