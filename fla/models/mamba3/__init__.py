
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mamba3.configuration_mamba3 import Mamba3Config
from fla.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM, Mamba3Model

AutoConfig.register(Mamba3Config.model_type, Mamba3Config, exist_ok=True)
AutoModel.register(Mamba3Config, Mamba3Model, exist_ok=True)
AutoModelForCausalLM.register(Mamba3Config, Mamba3ForCausalLM, exist_ok=True)


__all__ = ['Mamba3Config', 'Mamba3ForCausalLM', 'Mamba3Model']
