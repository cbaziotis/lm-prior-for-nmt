# 3M - Base
python sent_lm.py --config ../../configs/rnn/prior.lm_news_en_rnn.yaml  --name prior.lm_news_en_3M_rnn  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_de_rnn.yaml  --name prior.lm_news_de_3M_rnn  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_tr_rnn.yaml  --name prior.lm_news_tr_3M_rnn  --device cuda


# 3M - Big
python sent_lm.py --config ../../configs/rnn/prior.lm_news_en_rnn.yaml  --name prior.lm_news_en_3M_rnn_big batch_tokens=10000 model.emb_size=1024 model.rnn_size=2048 model.emb_dropout=0.3 model.rnn_dropout=0.3  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_de_rnn.yaml  --name prior.lm_news_de_3M_rnn_big batch_tokens=10000 model.emb_size=1024 model.rnn_size=2048 model.emb_dropout=0.3 model.rnn_dropout=0.3  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_tr_rnn.yaml  --name prior.lm_news_tr_3M_rnn_big batch_tokens=10000 model.emb_size=1024 model.rnn_size=2048 model.emb_dropout=0.3 model.rnn_dropout=0.3  --device cuda


# 30M - Base
python sent_lm.py --config ../../configs/rnn/prior.lm_news_en_rnn.yaml  --name prior.lm_news_en_30M_rnn data.train_path=../datasets/mono/priors/news.en.2014-2017.pp.30M.train  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_de_rnn.yaml  --name prior.lm_news_de_30M_rnn data.train_path=../datasets/mono/priors/news.de.2014-2017.pp.30M.train  --device cuda

# 30M - Big
python sent_lm.py --config ../../configs/rnn/prior.lm_news_en_rnn.yaml  --name prior.lm_news_en_30M_rnn_big data.train_path=../datasets/mono/priors/news.en.2014-2017.pp.30M.train batch_tokens=10000 model.emb_size=1024 model.rnn_size=2048 model.emb_dropout=0.3 model.rnn_dropout=0.3  --device cuda
python sent_lm.py --config ../../configs/rnn/prior.lm_news_de_rnn.yaml  --name prior.lm_news_de_30M_rnn_big data.train_path=../datasets/mono/priors/news.de.2014-2017.pp.30M.train batch_tokens=10000 model.emb_size=1024 model.rnn_size=2048 model.emb_dropout=0.3 model.rnn_dropout=0.3  --device cuda
