from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * (256 // 4)


#prompts = ["Once upon a time," ]

sampling_params = SamplingParams(temperature=0, top_p=1)

llm = LLM(model="/data/llm/checkpoints/llama-hf/llama-7b")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
