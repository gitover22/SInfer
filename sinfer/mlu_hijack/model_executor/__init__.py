# hijack sinfer layers
import sinfer.mlu_hijack.model_executor.layers.activation
import sinfer.mlu_hijack.model_executor.layers.feed_forward
import sinfer.mlu_hijack.model_executor.layers.layernorm
import sinfer.mlu_hijack.model_executor.layers.linear
import sinfer.mlu_hijack.model_executor.layers.rotary_embedding
import sinfer.mlu_hijack.model_executor.layers.logits_processor
import sinfer.mlu_hijack.model_executor.layers.quantization.weightonly
import sinfer.mlu_hijack.model_executor.layers.quantization.smoothquant
import sinfer.mlu_hijack.model_executor.layers.ops.rand
import sinfer.mlu_hijack.model_executor.layers.ops.sample

# hijack sinfer models
import sinfer.mlu_hijack.model_executor.models
import sinfer.mlu_hijack.model_executor.models.baichuan
import sinfer.mlu_hijack.model_executor.models.chatglm
import sinfer.mlu_hijack.model_executor.models.gpt_neox
import sinfer.mlu_hijack.model_executor.models.llama
import sinfer.mlu_hijack.model_executor.models.mixtral
import sinfer.mlu_hijack.model_executor.models.qwen
import sinfer.mlu_hijack.model_executor.models.qwen2
import sinfer.mlu_hijack.model_executor.models.qwen2_moe
import sinfer.mlu_hijack.model_executor.models.falcon

# hijack sinfer model loader
import sinfer.mlu_hijack.model_executor.model_loader.loader
