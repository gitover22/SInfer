import torch
import time
import statistics
import pandas as pd
import numpy as np
import subprocess
import json
import os
from datetime import datetime
from sinfer.logger import init_logger
from sinfer.Intercept.mlu_definitions import check_latency_debug_with_device
from sinfer.model_executor.layers.quantization import QUANTIZATION_METHODS

logger = init_logger(__name__)

class LLMMetric:
    '''
    用于收集和记录模型推理过程中产生的性能指标
    '''
    def __init__(self)->None:
        self.batch_size_list = []
        self.context_latency_list = []
        self.e2e_latency_list = []
        self.per_token_latency_list = [ [] ]
        self.per_token_latency_device_list = [ [] ]

    # 通过 GPU 同步后返回当前时间，用于测量精确的 GPU 操作时间
    def get_mlu_cost_time(self):
        torch.cuda.synchronize()
        return time.time()

    def is_prefill_stage(self):
        return len(self.per_token_latency_list[-1]) == 0

    def update_step_latency(self,step_latency):
        self.per_token_latency_list[-1].append(step_latency)

    def update_step_latency_device(self, step_latency):
        self.per_token_latency_device_list[-1].append(step_latency)

    # 将新的批量大小和端到端延迟加入记录列表
    def add_metrics(self,batch_size,e2e_latency)->None:
        self.batch_size_list.append(batch_size)
        self.e2e_latency_list.append(e2e_latency)
        self.per_token_latency_list.append([]) # new iter
        self.per_token_latency_device_list.append([])

    # 根据量化方法和模型配置获取权重的精度描述
    def get_weight_dtype_str(self, model_path, model_dtype, quantization) -> str:
        # get weight dtype based on quantization config if exists
        if quantization is not None:
            quant_method = QUANTIZATION_METHODS[quantization]
            # combine the model path with the quantization config file name
            quant_config_paths = quant_method.get_config_filenames()
            # if there are multiple quantization config files, return the first one existed
            for quant_config_path in quant_config_paths:
                quant_config_path = os.path.join(model_path, quant_config_path)
                # check if the quantization config file exists
                if not os.path.exists(quant_config_path):
                    continue
                with open(quant_config_path, 'r') as f:
                    quant_config = json.load(f)
                    quant_config = quant_method.from_config(quant_config)
                    # for smoothquant and weightonly, return the quantization name with the weight bits
                    if quant_method == QUANTIZATION_METHODS["smoothquant"] or quant_method == QUANTIZATION_METHODS["weightonly"]:
                        return "{}-int{}".format(quant_config.get_name(), quant_config.weight_bits)
                    else:
                        # for other quantization methods, return the quantization name
                        return quant_config.get_name()
            # if the quantization config file does not exist, just return the quanization name
            return quant_config_path.get_name()
        else:
            # remove the prefix of model dtype from torch config
            return str(model_dtype).split(".")[-1]
    
    # 将收集的性能数据写入 CSV 文件
    def to_csv(self, filename: str) -> None:
        df = pd.DataFrame(self.metrics_data)
        df_mean = df.mean().round(3)
        header = ["datetime", "model",
                  "weight dtype", self.batch_size_name, 
                  "input len", "output len", "tp", 
                  self.context_latency_name, self.per_token_latency_name]
        data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model,
                self.weight_dtype_str, int(self.metrics_data[self.batch_size_name][0]), 
                self.input_len, self.output_len, self.tp,
                df_mean[self.context_latency_name], df_mean[self.per_token_latency_name]]
        if check_latency_debug_with_device():
            header += [self.context_latency_device_name, self.per_token_latency_device_name]
            data += [df_mean[self.context_latency_device_name], df_mean[self.per_token_latency_device_name]]
        data_dict = dict(zip(header, data))
        df_csv = pd.DataFrame(data_dict, index=[0])
        append = False
        if os.path.isfile(filename):
            try:
                df_old = pd.read_csv(filename)
                append = (df_old.columns.tolist() == header)
            except Exception as e:
                logger.info(f"Existing {filename} failed to be read and will be overwritten")
        if append:
            df_csv.to_csv(filename, mode='a', header=False, index=False)
            logger.info(f"Metric appended to existing {filename}")
        else:
            df_csv.to_csv(filename, index=False)
            logger.info(f"Metric written to {filename}")

    # 计算和组织推理性能指标，包括上下文延迟、端到端延迟、吞吐量和 FLOPS 等
    def calc_metric(self,model,model_dtype,metrics_idx_start,only_average,input_len,output_len,tp_nums,quantization,dump_info=None)->None:
        keep_digits = 2
        metrics_idx_end = len(self.per_token_latency_list) - 1 # without last []
        idx_range = range(metrics_idx_start, metrics_idx_end)

        # specify entries to write to csv
        self.batch_size_name = "batch size"
        self.input_len = input_len
        self.output_len = output_len
        self.tp = tp_nums
        self.model = model
        self.context_latency_name = "context latency(ms)"
        self.per_token_latency_name = "per token latency(ms)"
        self.context_latency_device_name = "context latency device(ms)"
        self.per_token_latency_device_name = "per token latency device(ms)"
        self.weight_dtype_str = self.get_weight_dtype_str(model, model_dtype, quantization)

        metrics_data = [
            (self.batch_size_name,[ int(self.batch_size_list[i]) for i in idx_range]
            ), 
            (self.context_latency_name,[ round(1000 * self.per_token_latency_list[i][0],keep_digits) for i in idx_range]
            ),
            (self.per_token_latency_name,[   0.0 if len(self.per_token_latency_list[i]) == 1 else \
                                         round((sum(self.per_token_latency_list[i][1:])/float(len(self.per_token_latency_list[i][1:])))*1000,keep_digits) \
                                         for i in idx_range]
            ),
            ("e2e latency(ms)",[ round( 1000* self.e2e_latency_list[i],keep_digits) for i in idx_range]
            ),
            ("e2e throughput(tokens/s)",[  round(((len(self.per_token_latency_list[i]))/self.e2e_latency_list[i])*self.batch_size_list[i],keep_digits) \
                                         for i in idx_range],
            ),
            ("decoder throughput(tokens/s)",[ 0.0 if len(self.per_token_latency_list[i]) == 1 else \
                                            round((self.batch_size_list[i] * len(self.per_token_latency_list[i][1:])/float(sum(self.per_token_latency_list[i][1:]))),keep_digits) \
                                            for i in idx_range]
            )
        ]
        insert_latency_device = check_latency_debug_with_device()
        if insert_latency_device:
            context_latency_device = [round(self.per_token_latency_device_list[i][0], keep_digits)
                                      for i in idx_range]
            per_token_latency_device = [round(statistics.fmean(self.per_token_latency_device_list[i][1:]), 
                                              keep_digits) for i in idx_range]
            metrics_data.insert(3, (self.context_latency_device_name, context_latency_device))
            metrics_data.insert(4, (self.per_token_latency_device_name, per_token_latency_device))

        self.metrics_data = dict(metrics_data)
        self.to_csv("output.csv")
        
        # Print
        df = pd.DataFrame(self.metrics_data)
        df.loc["Average(" + str(metrics_idx_end-metrics_idx_start) + "iters)"] = df.mean().round(keep_digits)
        if only_average:
           df = pd.DataFrame([df.iloc[-1]], columns=df.columns)
        df["batch size"] = df["batch size"].astype(int)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        print("********************************* Test Info****************************")
        print("Generation Config input len:{} output len:{} tp_nums:{} quantization:{}".format(input_len,output_len,tp_nums,quantization))
        
        if dump_info and insert_latency_device:
            dump_info.init_param(batch_size=self.metrics_data['batch size'][0], 
                                 input_len=input_len, output_len=output_len, 
                                 context_latency_device=np.mean(self.metrics_data['context latency device(ms)']),
                                 generate_latency_device=np.mean(self.metrics_data['per token latency device(ms)']))
            dump_info.dump()
  
        print("*************************Performance Info******************************")
        print(df.to_string())              
        if insert_latency_device :            
            context_latency = np.mean(self.metrics_data['context latency device(ms)'])
            generate_latency = np.mean(self.metrics_data['per token latency device(ms)'])
            
            millisecond2second_unit = 1000
            if dump_info and dump_info.has_information_dump():
                dump_info.dump_performance_info()
            else:
                context_tflops_per_second = 0
                decoder_tflops_per_second = 0
                flops2Tflops = 1000 * 1000 * 1000 * 1000
                context_tflops = dump_info.flops_info.context_flops / flops2Tflops
                decoder_tflops = dump_info.flops_info.decoder_flops / flops2Tflops
                if not context_latency:
                    logger.warning("context_latency is 0, context_tflops_per_second unable to output correctly")
                else:
                    context_tflops_per_second = context_tflops / (context_latency / millisecond2second_unit)

                if not generate_latency:
                    logger.warning("generate_latency is 0, decoder_tflops_per_second unable to output correctly")
                else:
                    decoder_tflops_per_second = decoder_tflops / (generate_latency / millisecond2second_unit)
                print("Context tflops:  {} Tflops".format(context_tflops))
                print("Generate tflops: {} Tflops".format(decoder_tflops))
                print("Context tflops_per_second:  {} Tflops/s".format(context_tflops_per_second))
                print("Generate tflops_per_second: {} Tflops/s".format(decoder_tflops_per_second))
                df["context_tflops"] = context_tflops
                df["decoder_tflops"] = decoder_tflops
                df["context_tflops_per_second"] = context_tflops_per_second
                df["decoder_tflops_per_second"] = decoder_tflops_per_second
                if (not context_tflops) or (not decoder_tflops):
                    logger.warning("the flops is 0, Please check if the model file is correctly parsed!!!!!!!!!")
        print("***********************************************************************")