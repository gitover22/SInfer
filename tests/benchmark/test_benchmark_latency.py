
import numpy as np
from sinfer import LLM, SamplingParams
import os
import pandas as pd

def test_generating_csv():
    '''
    test generating csv
    '''
    # contents of this test is brought from benchmark_latency.py

    csv_file = "output.csv"
    if os.path.isfile(csv_file):
        os.remove("output.csv")
    assert not os.path.isfile(csv_file)

    os.environ['SINFER_LATENCY_DEBUG'] = "1"
    model_path = "/data/sinfer/sq_per_tensor_per_channel/Llama-2-7b-hf"
    tp = 1
    batch_size = 4
    input_len = 128
    output_len = 5
    quantization = "smoothquant"
    llm = LLM(model=model_path,
              tokenizer=model_path,
              quantization=quantization,
              tensor_parallel_size=tp,
              trust_remote_code=True,
              enforce_eager=True)
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=output_len,
    )
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(batch_size,
                                                     input_len))
    dummy_prompt_token_ids = dummy_prompt_token_ids.tolist()
    llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                 sampling_params=sampling_params,
                 use_tqdm=False)
    llm.get_metrics(0, # args.num_iters_warmup,
                    False, #args.only_average,
                    input_len, #args.input_len,
                    output_len, #args.output_len,
                    tp, #args.tensor_parallel_size,
                    quantization, #args.quantization
                    llm.dump_info)
    assert os.path.isfile(csv_file)
    df = pd.read_csv(csv_file)
    assert df['batch size'].item() == batch_size
    assert df['model'].item() == model_path
    assert df['input len'].item() == input_len
    assert df['output len'].item() == output_len
    assert df['tp'].item() == tp
    assert df['weight dtype'].item() == "SmoothQuant-int8"
    os.remove(csv_file)