from transformers import PretrainedConfig


class MinjaLMConfig(PretrainedConfig):
    model_type = "minja-lm"

    def __init__(self, vocab_size=32000, n_embd=128, n_layer=2, n_head=2, block_size=16, **kwargs):

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        super().__init__(**kwargs)
