from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import MinjaLMConfig
from .modeling import MinjaLM


AutoConfig.register("minja-lm", MinjaLMConfig)
AutoModelForCausalLM.register(MinjaLMConfig, MinjaLM)
