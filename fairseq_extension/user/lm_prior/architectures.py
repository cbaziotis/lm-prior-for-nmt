from fairseq.models import register_model_architecture
from fairseq.models.transformer import base_architecture
from fairseq.models.transformer_lm import base_lm_architecture


@register_model_architecture("transformer", "paper_transformer_mt")
def transformer_mt_base(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.dropout = getattr(args, "dropout", 0.3)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before",
                                            True)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before",
                                            True)

    base_architecture(args)


@register_model_architecture("transformer_lm", "paper_transformer_lm")
def transformer_lm_big(args):

    args.dropout = getattr(args, "dropout", 0.3)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before",
                                            True)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "paper_transformer_lm_small")
def paper_transformer_lm(args):
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before",
                                            True)
    base_lm_architecture(args)
