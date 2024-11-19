import torch


def make_context(
    tokenizer,
    query,
    history,
    system,
    max_input_length,
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    '''
    tokenize one text context to tokenized id
    args:
        tokenizer: model tokenizer
        query: current text context
        history: history text context
        system: system prompt
        max_input_length: max input length of tokenized id
        chat_format: chat format, only accept chatml and raw
    '''
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            '''
            tokensize string
            '''
            return (f"{role}\n{content}", tokenizer.encode(
                role,
                allowed_special=set(),
            ) + nl_tokens + tokenizer.encode(
                content,
                allowed_special=set(),
            ))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}")

            current_context_size = (len(system_tokens) + len(next_context_tokens) + len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (nl_tokens + im_start_tokens + _tokenize_str("user", query)[1] + im_end_tokens + nl_tokens +
                           im_start_tokens + tokenizer.encode("assistant") + nl_tokens)
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    # truncate to max_input_length, truncate from the front
    return raw_text, context_tokens[-max_input_length:]


def prepare_inputs(batch_input_texts,
                   tokenizer,
                   model_name,
                   model_version,
                   test_token_num,
                   eval_task='summarize',
                   add_special_tokens=True):
    '''
    tokenize batch input texts into tokenized id.
    args:
        batch_input_texts: batch input text, also named batched prompt
        tokenizer: model tokenizer
        model_name: model name
        model_version: model version
        test_token_num: batch size, also named prompt number
        eval_task: eval task
        add_special_tokens: whether to add_special_tokens, default True
    '''
    batch_size = len(batch_input_texts)
    append_str = ' TL;DR: ' if eval_task == 'summarize' else ''
    batch_input_ids = []
    for i in range(batch_size):
        curr_text = batch_input_texts[i] + append_str
        curr_text = curr_text.strip().replace(" n't", "n't")

        # The below lines are used to be compatible with the original code
        if model_name == 'ChatGLMForCausalLM' and model_version in ['chatglm2', 'chatglm3']:
            input_ids = tokenizer.encode(curr_text, return_tensors='pt').squeeze(0)
            input_ids = input_ids[:test_token_num]
        elif model_name == 'QWenForCausalLM':
            # use make_content to generate prompt
            system_prompt = "You are a useful assistant, please directly output the corresponding " + \
                "summary according to the article entered by the user."
            _, input_id_list = make_context(
                tokenizer=tokenizer,
                query=curr_text,
                history=[],
                system=system_prompt,
                max_input_length=test_token_num,
            )
            input_ids = torch.tensor(input_id_list)
        else:
            input_ids = tokenizer.encode(curr_text,
                                         return_tensors='pt',
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=test_token_num).squeeze(0)

        batch_input_ids.append(input_ids)
    return batch_input_ids
