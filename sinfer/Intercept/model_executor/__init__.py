# hijack sinfer layers
import sinfer.Intercept.model_executor.layers.activation
import sinfer.Intercept.model_executor.layers.feed_forward
import sinfer.Intercept.model_executor.layers.layernorm
import sinfer.Intercept.model_executor.layers.linear
import sinfer.Intercept.model_executor.layers.rotary_embedding
import sinfer.Intercept.model_executor.layers.logits_processor
import sinfer.Intercept.model_executor.layers.quantization.weightonly
import sinfer.Intercept.model_executor.layers.quantization.smoothquant
import sinfer.Intercept.model_executor.layers.ops.rand
import sinfer.Intercept.model_executor.layers.ops.sample

# hijack sinfer models
import sinfer.Intercept.model_executor.models
import sinfer.Intercept.model_executor.models.baichuan
import sinfer.Intercept.model_executor.models.chatglm
import sinfer.Intercept.model_executor.models.gpt_neox
import sinfer.Intercept.model_executor.models.llama
import sinfer.Intercept.model_executor.models.mixtral
import sinfer.Intercept.model_executor.models.qwen
import sinfer.Intercept.model_executor.models.qwen2
import sinfer.Intercept.model_executor.models.qwen2_moe
import sinfer.Intercept.model_executor.models.falcon

# hijack sinfer model loader
import sinfer.Intercept.model_executor.model_loader.loader
