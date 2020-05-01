def experiment(pair, lamda, tau, objective, nhid, smoothing=True):
    cmd = "python nmt_prior.py"
    cmd += f" --config ../../configs/acl_transformer/trans.{pair}_prior.yaml"
    cmd += f" losses.prior.weight={lamda}"
    cmd += f" losses.prior.tau={tau}"
    cmd += f" losses.prior.objective={objective}"
    cmd += f" model.nhid={nhid}"

    name = f" --name trans.{pair}" \
           f"_prior_w={lamda}_tau={tau}_obj={objective}_nhid={nhid}"

    if smoothing:
        cmd += " losses.mt.smoothing=0.1"
        name += "_ls"

    cmd += name

    return cmd


_pair = "deen"
_prior = "../../checkpoints/prior.lm_news_en_30M_trans_best.pt"
with open("experiments_sensitivity.sh", "w") as f:
    for _lamda in [0.02, 0.1, 0.2, 0.5, 1]:
        for _tau in [1, 2, 5]:
            for _nhid in [1024]:
                for _obj in ["kl", "rkl"]:
                    for _ls in [True, False]:
                        _cmd = experiment(pair=_pair,
                                          lamda=_lamda,
                                          tau=_tau,
                                          objective=_obj,
                                          nhid=_nhid,
                                          smoothing=_ls)
                        _cmd += f" data.prior_path={_prior}"
                        _cmd += " --device cuda"
                        f.write(_cmd + "\n")
