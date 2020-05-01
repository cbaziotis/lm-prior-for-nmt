from modules.trainer import Trainer


class SentLMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = self.config["model"].get("type", "rnn")

    def process_batch(self, inputs, labels, lengths):
        losses = dict()
        predictions = self.model(inputs, lengths)
        losses["lm"] = self.cross_entropy_loss(predictions["logits"], labels)
        return losses, predictions

    def get_vocab(self):
        return self.valid_loader.dataset.vocab
