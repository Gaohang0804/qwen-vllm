from vllm_wrapper import vLLMWrapper

model = "Qwen/Qwen1.5-0.5B-Chat"

vllm_model = vLLMWrapper(model,
                         quantization='gptq',
                         dtype="float16",
                         tensor_parallel_size=1,
                         gpu_memory_utilization=0.6)

history = None
while True:
    Q = input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history = history[:20]
