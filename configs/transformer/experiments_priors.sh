# 3M - Base
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_en_trans.yaml  --device cuda  --name prior.lm_news_en_3M_trans
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_de_trans.yaml  --device cuda  --name prior.lm_news_de_3M_trans
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_tr_trans.yaml  --device cuda  --name prior.lm_news_tr_3M_trans

# 3M - Big
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_en_trans.yaml  --device cuda  --name prior.lm_news_en_3M_trans_big batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_de_trans.yaml  --device cuda  --name prior.lm_news_de_3M_trans_big batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_tr_trans.yaml  --device cuda  --name prior.lm_news_tr_3M_trans_big batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3

# 30M - Base
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_en_trans.yaml  --device cuda  --name prior.lm_news_en_30M_trans data.train_path=../datasets/mono/priors/news.en.2014-2017.pp.30M.train
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_de_trans.yaml  --device cuda  --name prior.lm_news_de_30M_trans data.train_path=../datasets/mono/priors/news.de.2014-2017.pp.30M.train

# 30M - Big
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_en_trans.yaml  --device cuda  --name prior.lm_news_en_30M_trans_big data.train_path=../datasets/mono/priors/news.en.2014-2017.pp.30M.train batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3
python mono_sent.py --config ../configs/acl_transformer/prior.lm_news_de_trans.yaml  --device cuda  --name prior.lm_news_de_30M_trans_big data.train_path=../datasets/mono/priors/news.de.2014-2017.pp.30M.train batch_tokens=12000 model.emb_size=1024 model.nhid=4096 model.nhead=16 model.dropout=0.3
