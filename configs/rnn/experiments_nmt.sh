python nmt_prior.py --config ../../configs/rnn/rnn.deen_base.yaml --name final.rnn.deen_base
python nmt_prior.py --config ../../configs/rnn/rnn.deen_base.yaml losses.mt.smoothing=0.1 --name final.rnn.deen_base_ls
python nmt_prior.py --config ../../configs/rnn/rnn.deen_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.deen_prenorm
python nmt_prior.py --config ../../configs/rnn/rnn.deen_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.deen_postnorm
python nmt_prior.py --config ../../configs/rnn/rnn.deen_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.deen_prenorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.deen_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.deen_postnorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.deen_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.deen_prior_3M_kl
python nmt_prior.py --config ../../configs/rnn/rnn.deen_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.deen_prior_3M_kl_ls
python nmt_prior.py --config ../../configs/rnn/rnn.ende_base.yaml --name final.rnn.ende_base
python nmt_prior.py --config ../../configs/rnn/rnn.ende_base.yaml losses.mt.smoothing=0.1 --name final.rnn.ende_base_ls
python nmt_prior.py --config ../../configs/rnn/rnn.ende_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt --name final.rnn.ende_prenorm
python nmt_prior.py --config ../../configs/rnn/rnn.ende_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt --name final.rnn.ende_postnorm
python nmt_prior.py --config ../../configs/rnn/rnn.ende_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.ende_prenorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.ende_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.ende_postnorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.ende_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt --name final.rnn.ende_prior_3M_kl
python nmt_prior.py --config ../../configs/rnn/rnn.ende_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_de_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.ende_prior_3M_kl_ls
python nmt_prior.py --config ../../configs/rnn/rnn.entr_base.yaml --name final.rnn.entr_base
python nmt_prior.py --config ../../configs/rnn/rnn.entr_base.yaml losses.mt.smoothing=0.1 --name final.rnn.entr_base_ls
python nmt_prior.py --config ../../configs/rnn/rnn.entr_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt --name final.rnn.entr_prenorm
python nmt_prior.py --config ../../configs/rnn/rnn.entr_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt --name final.rnn.entr_postnorm
python nmt_prior.py --config ../../configs/rnn/rnn.entr_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.entr_prenorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.entr_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.entr_postnorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.entr_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt --name final.rnn.entr_prior_3M_kl
python nmt_prior.py --config ../../configs/rnn/rnn.entr_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_tr_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.entr_prior_3M_kl_ls
python nmt_prior.py --config ../../configs/rnn/rnn.tren_base.yaml --name final.rnn.tren_base
python nmt_prior.py --config ../../configs/rnn/rnn.tren_base.yaml losses.mt.smoothing=0.1 --name final.rnn.tren_base_ls
python nmt_prior.py --config ../../configs/rnn/rnn.tren_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.tren_prenorm
python nmt_prior.py --config ../../configs/rnn/rnn.tren_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.tren_postnorm
python nmt_prior.py --config ../../configs/rnn/rnn.tren_fusion.yaml model.decoding.fusion=prenorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.tren_prenorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.tren_fusion.yaml model.decoding.fusion=postnorm data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.tren_postnorm_ls
python nmt_prior.py --config ../../configs/rnn/rnn.tren_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt --name final.rnn.tren_prior_3M_kl
python nmt_prior.py --config ../../configs/rnn/rnn.tren_prior.yaml losses.prior.objective=kl data.prior_path=../checkpoints/prior.lm_news_en_3M_rnn_big_best.pt losses.mt.smoothing=0.1 --name final.rnn.tren_prior_3M_kl_ls
