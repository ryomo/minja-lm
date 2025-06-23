from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import MinjaLMConfig
from .modeling import MinjaLM


# Register for AutoClass support
AutoConfig.register("minja-lm", MinjaLMConfig)
AutoModelForCausalLM.register(MinjaLMConfig, MinjaLM)

# Register for auto_map in config.json when uploading to Hub
MinjaLMConfig.register_for_auto_class()
MinjaLM.register_for_auto_class("AutoModelForCausalLM")
