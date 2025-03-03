CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "/home/liyanhao/FActScore/model/DeepSeek-R1-Distill-Qwen-7B" \
		--served-model-name "DeepSeek-R1-Distill-Qwen-7B" \
        --max_model_len 8192 \
        --gpu_memory_utilization 0.9 \
        --tensor-parallel-size 4 \
        --port 8222

