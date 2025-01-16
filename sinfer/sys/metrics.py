import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Counter as CollectionsCounter
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
from prometheus_client import (REGISTRY, Counter, Gauge, Histogram, Info,
                               disable_created_metrics)
import csv
from sinfer.logger import init_logger

logger = init_logger(__name__)

disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
class Metrics:
    labelname_finish_reason = "finished_reason"

    def __init__(self, labelnames: List[str], max_model_len: int):
        # Unregister any existing SInfer collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "sinfer" in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_cache_config = Info(
            name='sinfer:cache_config',
            documentation='information of cache_config')

        # System stats
        #   Scheduler State
        self.gauge_scheduler_running = Gauge(
            name="sinfer:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = Gauge(
            name="sinfer:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames)
        self.gauge_scheduler_swapped = Gauge(
            name="sinfer:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames)
        #   KV Cache Usage in %
        self.gauge_gpu_cache_usage = Gauge(
            name="sinfer:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        self.gauge_cpu_cache_usage = Gauge(
            name="sinfer:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)

        # Iteration stats
        self.counter_prompt_tokens = Counter(
            name="sinfer:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        self.counter_generation_tokens = Counter(
            name="sinfer:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        self.histogram_time_to_first_token = Histogram(
            name="sinfer:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ])
        self.histogram_time_per_output_token = Histogram(
            name="sinfer:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5
            ])

        # Request stats
        #   Latency
        self.histogram_e2e_time_request = Histogram(
            name="sinfer:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        #   Metadata
        self.histogram_num_prompt_tokens_request = Histogram(
            name="sinfer:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = Histogram(
            name="sinfer:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_best_of_request = Histogram(
            name="sinfer:request_params_best_of",
            documentation="Histogram of the best_of request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_n_request = Histogram(
            name="sinfer:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.counter_request_success = Counter(
            name="sinfer:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])

        # Deprecated in favor of sinfer:prompt_tokens_total
        self.gauge_avg_prompt_throughput = Gauge(
            name="sinfer:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
        )
        # Deprecated in favor of sinfer:generation_tokens_total
        self.gauge_avg_generation_throughput = Gauge(
            name="sinfer:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
        )


# end-metrics-definitions
def build_1_2_5_buckets(max_value: int):
    """
    Builds a list of buckets with increasing powers of 10 multiplied by 
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1

@dataclass
class Stats:
    now: float

    # System stats (should have _sys suffix)
    #   Scheduler State
    num_running_sys: int
    num_ready_sys: int
    num_blocked_sys: int
    #   KV Cache Usage in %
    mlu_cache_usage_sys: float
    cpu_cache_usage_sys: float

    # Iteration stats (should have _iter suffix)
    num_prompt_tokens_iter: int
    num_generation_tokens_iter: int
    time_to_first_tokens_iter: List[float]
    time_per_output_tokens_iter: List[float]

    # Request stats (should have _requests suffix)
    #   Latency
    time_e2e_requests: List[float]
    #   Metadata
    num_prompt_tokens_requests: List[int]
    num_generation_tokens_requests: List[int]
    best_of_requests: List[int]
    n_requests: List[int]
    finished_reason_requests: List[str]

class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> Dict[str, str]:
        ...


class StatLogger:
    def __init__(self, local_interval: float, labels: Dict[str, str],
                 max_model_len: int) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.time()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

        # Prometheus metrics
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()),
                               max_model_len=max_model_len)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        if type == "cache_config":
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        # System state data
        self._log_gauge(self.metrics.gauge_scheduler_running,
                        stats.num_running_sys)
        self._log_gauge(self.metrics.gauge_scheduler_swapped,
                        stats.num_blocked_sys)
        self._log_gauge(self.metrics.gauge_scheduler_waiting,
                        stats.num_ready_sys)
        self._log_gauge(self.metrics.gauge_gpu_cache_usage,
                        stats.mlu_cache_usage_sys)
        self._log_gauge(self.metrics.gauge_cpu_cache_usage,
                        stats.cpu_cache_usage_sys)

        # Iteration level data
        self._log_counter(self.metrics.counter_prompt_tokens,
                          stats.num_prompt_tokens_iter)
        self._log_counter(self.metrics.counter_generation_tokens,
                          stats.num_generation_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_to_first_token,
                            stats.time_to_first_tokens_iter)
        self._log_histogram(self.metrics.histogram_time_per_output_token,
                            stats.time_per_output_tokens_iter)

        # Request level data
        # Latency
        self._log_histogram(self.metrics.histogram_e2e_time_request,
                            stats.time_e2e_requests)
        # Metadata
        finished_reason_counter = CollectionsCounter(
            stats.finished_reason_requests)
        self._log_counter_labels(self.metrics.counter_request_success,
                                 finished_reason_counter,
                                 Metrics.labelname_finish_reason)
        self._log_histogram(self.metrics.histogram_num_prompt_tokens_request,
                            stats.num_prompt_tokens_requests)
        self._log_histogram(
            self.metrics.histogram_num_generation_tokens_request,
            stats.num_generation_tokens_requests)
        self._log_histogram(self.metrics.histogram_n_request, stats.n_requests)
        self._log_histogram(self.metrics.histogram_best_of_request,
                            stats.best_of_requests)

    def _log_gauge(self, gauge: Gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter: Counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(self, counter: Counter, data: CollectionsCounter,
                            label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram: Histogram,
                       data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on
        # the SInfer side. Moving forward, we should use counters like
        # counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the
        # grafana/prometheus side. See
        # https://github.com/sinfer-project/sinfer/pull/2316#discussion_r1464204666
        self.metrics.gauge_avg_prompt_throughput.labels(
            **self.labels).set(prompt_throughput)
        self.metrics.gauge_avg_generation_throughput.labels(
            **self.labels).set(generation_throughput)

    def log(self, stats: Stats) -> None:
        self._log_prometheus(stats)  # Log to prometheus.

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens_iter)
        self.num_generation_tokens.append(stats.num_generation_tokens_iter)

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now) or stats.time_e2e_requests \
                                                    or stats.time_to_first_tokens_iter :
            # Compute summary metrics for tracked stats (log them to promethus).
            prompt_throughput = self._get_throughput(self.num_prompt_tokens,
                                                    now=stats.now)
            generation_throughput = self._get_throughput(
                self.num_generation_tokens, now=stats.now)

            # Calculate averages for time-related metrics
            avg_time_e2e_requests = np.mean(stats.time_e2e_requests) if stats.time_e2e_requests else 0
            avg_time_to_first_tokens_iter = np.mean(stats.time_to_first_tokens_iter) if stats.time_to_first_tokens_iter else 0
            avg_time_per_output_tokens_iter = np.mean(stats.time_per_output_tokens_iter) if stats.time_per_output_tokens_iter else 0

            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            logger.info(
                "Avg prompt throughput: %.1f tokens/s, "
                "Avg generation throughput: %.1f tokens/s, "
                "Running: %d reqs, blocked: %d reqs, "
                "ready: %d reqs, MLU KV cache usage: %.1f%%, "
                "CPU KV cache usage: %.1f%%, "
                "Avg E2E latency: %.3f s, "
                "Avg time to first token: %.3f s, "
                "Avg time per output token: %.3f s",
                prompt_throughput,
                generation_throughput,
                stats.num_running_sys,
                stats.num_blocked_sys,
                stats.num_ready_sys,
                stats.mlu_cache_usage_sys * 100,    
                stats.cpu_cache_usage_sys * 100,
                avg_time_e2e_requests,
                avg_time_to_first_tokens_iter,
                avg_time_per_output_tokens_iter
            )

            from datetime import datetime

            current_time = datetime.now().strftime("%Y_%m_%d")
            csv_file_path = f'sys_metrics_{current_time}.csv'

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                # 写入文件头
                writer.writerow([
                    'index', 'generation_throughput',
                    'runningPool', 'readyPool',
                    'mlu_cache_usage', 'avg_time_e2e_requests',
                    'avg_time_to_first_tokens_iter', 'avg_time_per_output_tokens_iter'
                ])
                # writer.writerow([
                #     'index', 'prompt_throughput', 'generation_throughput',
                #     'runningPool', 'readyPool',
                #     'mlu_cache_usage', 'avg_time_e2e_requests',
                #     'avg_time_to_first_tokens_iter', 'avg_time_per_output_tokens_iter'
                # ])
                # 写入数据
                writer.writerow([
                    self._get_next_index(),  # 序号
                    # prompt_throughput,  # tokens/s
                    generation_throughput,  # tokens/s
                    stats.num_running_sys,  # reqs
                    stats.num_ready_sys,    # reqs
                    stats.mlu_cache_usage_sys * 100,  # %
                    avg_time_e2e_requests,      # s     
                    avg_time_to_first_tokens_iter,  # s
                    avg_time_per_output_tokens_iter # s
                ])

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now

    def _get_next_index(self) -> int:
        if not hasattr(self, '_current_index'):
            self._current_index = 0 
        else:
            self._current_index += 1
        return self._current_index