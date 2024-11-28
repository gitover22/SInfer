#
# todo: device_info.py文件没有实现
#

from sinfer.logger import init_logger
from sinfer.mlu_hijack.mlu_hijack_utils import get_is_gated, ModelConfig
import ctypes
import json
from sinfer.transformers_utils.config import get_config
from sinfer.entrypoints.llm import LLM
from sinfer.mlu_hijack.mlu_definitions import check_dump_cpu_info, check_dump_mlu_info
logger = init_logger(__name__)

class FlopsInfo(ctypes.Structure):
    _fields_ = [("context_flops", ctypes.c_double),
                ("decoder_flops", ctypes.c_double)]

class LLMDumpInfo:
    '''
    用于收集和处理模型运行过程中与性能相关的信息
    '''
    def __init__(self,
                 tensor_parallel_size=None,
                 dtype=None, kv_cache_dtype=None,
                 quantization=None,
                 model=None, batch_size=None,
                 input_len=None,
                 output_len=None,
                 trust_remote_code=None)->None:
        self.so_file = None
        self.dev_info = None
        self.cpu_info = None
        self.lib = None
        self.hfu_info = None
        self.flops_info = None
        self.ctypes_model_config = ModelConfig()
        self.io_efficiency = 0
        self.context_latency_device = 0
        self.generate_latency_device = 0

        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization = quantization
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self.model = model
        self.model_config = None

        try:
            from sinfer.mlu_hijack.device_info import get_info_inner
            self.so_file,self.dev_info,self.cpu_info,self.lib = get_info_inner(self.so_file, self.dev_info, self.cpu_info, self.lib)
        except:
            # logger.info("Cannot get device info")
            pass
    
    def init_param(self,
                   tensor_parallel_size=None,
                   dtype=None,
                   kv_cache_dtype=None,
                   quantization=None,
                   model=None,
                   batch_size=None,
                   input_len=None,
                   output_len=None,
                   trust_remote_code=None,
                   context_latency_device=None,
                   generate_latency_device=None):
        if tensor_parallel_size != None:
            self.tensor_parallel_size = tensor_parallel_size
        if dtype != None:
            self.dtype = dtype
        if kv_cache_dtype != None:
            self.kv_cache_dtype = kv_cache_dtype
        if quantization != None:
            self.quantization = quantization
        if model != None:
            self.model = model
        if batch_size != None:
            self.batch_size = batch_size
        if input_len != None:
            self.input_len = input_len
        if output_len != None:
            self.output_len = output_len
        if trust_remote_code != None:
            self.trust_remote_code = trust_remote_code
        if context_latency_device != None:
            self.context_latency_device = context_latency_device
        if generate_latency_device != None:
            self.generate_latency_device = generate_latency_device
        
        # 解析模型配置
        if self.model_config == None and self.model != None and self.trust_remote_code != None:
            self.model_config = get_config(self.model, self.trust_remote_code)
        
    def initialize_ctypes_model_config(self, model_cfg, tp_num, weight_dtype, kv_cache_dtype, quantization):    
        # prepare input
        self.ctypes_model_config.hidden_size     = model_cfg.hidden_size
        self.ctypes_model_config.vocab_size      = model_cfg.vocab_size
        
        possible_keys_ffn_size = [
            # chatglm3-6b-32k
            "ffn_hidden_size",
            # llama3-8b-hf
            "intermediate_size",
            # custom moe
            "moe_intermediate_size"
        ]
        possible_kv_heads = [
            # chatglm3-6b-32k
            "multi_query_group_num",
            # llama3-8b-hf
            "num_key_value_heads",
            # falcon-180B-chat
            "num_kv_heads",
        ]
        possible_num_attention_heads = [
            "num_attention_heads",
            "n_heads",
        ]
        for key in possible_keys_ffn_size:
            ffn_size = getattr(model_cfg, key, None)
            if ffn_size is not None:
                break
        if ffn_size is None:
            logger.warning("The model's config.json does not contain any of the following"
                        "keys to determine the ffn_size: "
                        f"{possible_keys_ffn_size}. ")
            
        for key in possible_num_attention_heads:
            num_attention_heads = getattr(model_cfg, key, None)
            if num_attention_heads is not None:
                break
        if num_attention_heads is None:
            logger.error("The model's config.json does not contain any of the following"
                        "keys to determine the num_attention_heads: "
                        f"{possible_num_attention_heads}. ")
        
        for key in possible_kv_heads:
            kv_heads = getattr(model_cfg, key, None)
            if kv_heads is not None:
                break

        if kv_heads is None:
            logger.warning("The model's config.json does not contain any of the following"
                        "keys to determine the kv_heads: "
                        f"{possible_kv_heads}, use num_attention_heads to replace")
            kv_heads = model_cfg.num_attention_heads
            
        self.ctypes_model_config.ffn_inner_size  =  ffn_size
        self.ctypes_model_config.layer_num       =  model_cfg.num_hidden_layers
        self.ctypes_model_config.head_num        =  num_attention_heads
        self.ctypes_model_config.head_size       =  self.ctypes_model_config.hidden_size / self.ctypes_model_config.head_num
        self.ctypes_model_config.head_num_kv     =  kv_heads
        self.ctypes_model_config.tp_num          =  tp_num
        self.ctypes_model_config.use_gated_ffn   =  get_is_gated()
        if hasattr(model_cfg, "num_experts") and model_cfg.num_experts is not None:
            self.ctypes_model_config.experts_num     =  model_cfg.num_experts
            self.ctypes_model_config.topk_num        =  model_cfg.num_experts_per_tok
        elif hasattr(model_cfg, "num_local_experts"):
            self.ctypes_model_config.experts_num     =  model_cfg.num_local_experts
            self.ctypes_model_config.topk_num        =  model_cfg.num_experts_per_tok
        else:
            self.ctypes_model_config.experts_num     =  0
        
        self.ctypes_model_config.use_causal_mask     =  True  # the flash attention is only use causal_mask in sinfer

        if weight_dtype == "atuo":
            self.ctypes_model_config.data_type = b'float16'
        else:
            self.ctypes_model_config.data_type = weight_dtype.encode('utf-8') 
            
        if quantization != None:
            with open(self.model + "/quantize_config.json", 'r') as file:
                config = json.load(file)      
            if config["quant_mode"] == "SmoothQuant":
                self.ctypes_model_config.smooth_quant_type == b"SmoothQuant"
            else:
                self.ctypes_model_config.smooth_quant_type = b'invalid'
            self.ctypes_model_config.filter_data_type = ("int" +  str(config['bits'])).encode('utf-8')
        else:
            self.ctypes_model_config.filter_data_type = self.ctypes_model_config.data_type
            
        if kv_cache_dtype == "auto":    
            self.ctypes_model_config.kv_cache_dtype      =  self.ctypes_model_config.data_type
        else:
            self.ctypes_model_config.kv_cache_dtype      =  kv_cache_dtype.encode('utf-8')


    def get_flops(self, bcfg, once_batch, input_seq_len, output_length, flops_info):
        self.batch_size = once_batch
        seq_len = input_seq_len
        hidden_size = bcfg.hidden_size
        voc_size = bcfg.vocab_size
        ffn_size = bcfg.ffn_inner_size
        layer_num = bcfg.layer_num
        out_seq = output_length
        seq_len_decode = seq_len + out_seq / 2
        r = bcfg.head_num / bcfg.head_num_kv
        bsh2 = self.batch_size * seq_len * hidden_size * hidden_size
        context_atn_pre = 2 * bsh2 + 4 * bsh2 / r
        context_atn_qk = 2 * self.batch_size * seq_len * seq_len * hidden_size
        context_atn_qkv = 2 * self.batch_size * seq_len * seq_len * hidden_size
        context_atn_post = 2 * self.batch_size * seq_len * hidden_size * hidden_size
        context_lm_head = 2 * self.batch_size * seq_len * hidden_size * voc_size
        context_ffn = 0
        bh2 = self.batch_size * hidden_size * hidden_size
        decode_atn_pre = 2 * bh2 + 4 * bh2 / r
        decode_atn_qk = 2 * self.batch_size * seq_len_decode * hidden_size
        decode_atn_qkv = 2 * self.batch_size * seq_len_decode * hidden_size
        decode_atn_post = 2 * self.batch_size * hidden_size * hidden_size
        decode_lm_head = 2 * self.batch_size * hidden_size * voc_size
        decode_ffn = 0
        coeffient = 6 if bcfg.use_gated_ffn else 4
        if bcfg.experts_num == 0:
            context_ffn = coeffient * self.batch_size * seq_len * hidden_size * ffn_size
            decode_ffn = coeffient * self.batch_size * hidden_size * ffn_size
        else:
            context_ffn = self.batch_size * seq_len * hidden_size * (coeffient * ffn_size * bcfg.topk_num + 2 * bcfg.experts_num)
            decode_ffn = self.batch_size * hidden_size * (coeffient * ffn_size * bcfg.topk_num + 2 * bcfg.experts_num)
        
        if bcfg.use_causal_mask:
            c = 0.5
            context_atn_qk *= c
            context_atn_qkv *= c

        flops_info.context_flops = context_lm_head
        flops_info.decoder_flops = decode_lm_head
        if bcfg.kv_cache_dtype != b"int8":
            flops_info.context_flops += (layer_num * (context_atn_qk + context_atn_qkv))
            flops_info.decoder_flops += (layer_num * (decode_atn_qk + decode_atn_qkv))
        else:
            flops_info.context_flops += (layer_num * (context_atn_qk + context_atn_qkv))
            flops_info.decoder_flops += (layer_num * (decode_atn_qk + decode_atn_qkv))
        
        if bcfg.smooth_quant_type == b"invalid": 
            flops_info.context_flops += (layer_num * (context_atn_pre + context_atn_post + context_ffn))
            flops_info.decoder_flops += (layer_num * (decode_atn_pre + decode_atn_post + decode_ffn))
        else:
            flops_info.context_flops += (layer_num * (context_atn_pre + context_atn_post + context_ffn))
            flops_info.decoder_flops += (layer_num * (decode_atn_pre + decode_atn_post + decode_ffn))
    
    # cpu信息采集
    def capture_cpu_info(self):
        if check_dump_cpu_info() and self.cpu_info:
            try:
                from sinfer.mlu_hijack.device_info import capture_cpu_info
                self.cpu_info = capture_cpu_info(self.cpu_info, my_rank=0)
            except:
                logger.info("Unsupport capture_cpu_info function")
    # cpu DRAM使用情况                   
    def memory_usage(self):
        if check_dump_cpu_info() and self.cpu_info:
            try:
                from sinfer.mlu_hijack.device_info import memory_usage
                self.cpu_info = memory_usage(self.cpu_info)
            except:
                logger.info("Unsupport memory_usage function")

    def analyze_perf_data(self, rank=0):
        try:
            from sinfer.mlu_hijack.device_info import analyze_perf_data
            analyze_perf_data(self.cpu_info, self.lib)
        except:
            logger.info("Cannot analyze perf data, no analyze_perf_data function")
            
    def get_decoder_io_efficiency(self, ctypes_model_config, lib, batch_size, input_len, output_len, generate_latency_device):
        try:
            from sinfer.mlu_hijack.device_info import get_decoder_io_efficiency
            self.io_efficiency = get_decoder_io_efficiency(ctypes_model_config, lib, batch_size, input_len, output_len, generate_latency_device)
        except: 
            logger.info("Unsupport io_efficiency get_decoder_io_efficiency function")        
            
    def get_device_output_info(self,
                               model_cfg,
                               batch_size,
                               input_seq_len,
                               output_length,
                               tp_num,
                               weight_dtype,
                               kv_cache_dtype,
                               quantization):
        self.initialize_ctypes_model_config(model_cfg, tp_num, weight_dtype, kv_cache_dtype, quantization)
        if check_dump_cpu_info() and self.so_file:
            self.analyze_perf_data()
        if check_dump_mlu_info() and self.lib:
            from sinfer.mlu_hijack.device_info import get_flops_inner, HFUInfo
            self.hfu_info = HFUInfo()
            get_flops_inner(self.ctypes_model_config, batch_size, input_seq_len, output_length, tp_num, self.hfu_info, self.lib)
            self.get_decoder_io_efficiency(self.ctypes_model_config,
                                           self.lib,
                                           self.batch_size,
                                           self.input_len,
                                           self.output_len,
                                           self.generate_latency_device)
        else:
            self.flops_info = FlopsInfo()
            self.get_flops(self.ctypes_model_config, batch_size, input_seq_len, output_length, self.flops_info)
            
    def has_information_dump(self):
        if self.dev_info and self.dev_info.so_file:
            return True
        return False
              
    def dump(self):
        self.get_device_output_info(self.model_config,
                                    self.batch_size,
                                    self.input_len,
                                    self.output_len,
                                    self.tensor_parallel_size,
                                    self.dtype,
                                    self.kv_cache_dtype,
                                    self.quantization)
        try:
            from sinfer.mlu_hijack.device_info import dump
            dump(LLM.dump_info)
        except:
            logger.info("Unsupport dump device/cpu information")
            
    def dump_performance_info(self):
        try:
            from sinfer.mlu_hijack.device_info import dump_information
            dump_information(LLM.dump_info)
        except:
            logger.info("Unsupport dump performance information")
        
    
