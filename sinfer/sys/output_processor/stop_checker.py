from typing import Callable, Optional

from transformers import PreTrainedTokenizer

from sinfer.sampling_params import SamplingParams
from sinfer.request import Request, RequestStatus


class StopChecker:
    # 检查以下内容：已发出eos token，已消耗最大token数，是否已发出停止字符串，或是否超过了最大模型长度。
    def __init__(self, max_model_len: int,
                 get_req_tokenizer: Callable[[Request],
                                                 PreTrainedTokenizer]):
        self.max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_req_tokenizer

    def maybe_stop_request(self, req: Request, new_char_count: int,
                            sampling_params: SamplingParams) -> None:
        """停止已完成的req
        new_char_count 是新增的字符数, 用于序列输出文本中新生成的token
        """

        # 检查是否已生成最小数量的token
        # 如果没有，则跳过停止字符串/token的检查
        if req.get_output_len() < sampling_params.min_tokens:
            return

        # 检查是否生成了EOS token
        if ((not sampling_params.ignore_eos)
                and req.get_last_token_id() == req.eos_token_id):
            req.status = RequestStatus.FINISHED_STOPPED
            return

        # 检查是否遇到了停止token: 假设每步生成一个token
        last_token_id = req.get_last_token_id()
        if last_token_id in sampling_params.stop_token_ids:
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                req.output_text = req.output_text[:-new_char_count]
            req.status = RequestStatus.FINISHED_STOPPED
            req.stop_reason = last_token_id
            return

        # 检查是否匹配到任何停止字符串
        stop_str = self._check_stop_strings(req, new_char_count,
                                            sampling_params)
        if stop_str is not None:
            req.status = RequestStatus.FINISHED_STOPPED
            req.stop_reason = stop_str
            return

        # 检查是否达到max_model_len
        if req.get_len() > self.max_model_len:
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return

        # 检查序列是否达到max_tokens.
        if req.get_output_len() == sampling_params.max_tokens:
            req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return

    @staticmethod
    def _check_stop_strings(req: Request, new_char_count: int,
                            sampling_params: SamplingParams) -> Optional[str]:
        """检查是否匹配到任何停止字符串，并相应地截断序列输出文本
        如果匹配到停止字符串, 则返回该字符串, 否则返回None
        """
        if not new_char_count:
            return None

        for stop_str in sampling_params.stop:
            stop_string_len = len(stop_str)
            # 避免搜索已搜索的文本
            stop_index = req.output_text.find(
                stop_str, -new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if sampling_params.include_stop_str_in_output:
                # 截断到停止字符串的末尾
                stop_index += stop_string_len
                if stop_index >= len(req.output_text):
                    # No truncation required.
                    return stop_str

            # 截断输出文本到停止字符串的起始或结束位置
            req.output_text = req.output_text[:stop_index]
            return stop_str
        return None
