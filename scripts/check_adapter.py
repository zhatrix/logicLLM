"""离线对比：不带 / 带 LoRA 时，模型对同一个问题的输出。训练后先跑这个，确认 adapter 真的生效再上线。"""
from mlx_lm import load, generate
from logicllm.agent.prompts import build_system, openai_tools
from logicllm.tools import all_specs

msgs = [{"role": "system", "content": build_system()},
        {"role": "user", "content": "提取收件信息：收件人王小明 13812345678 浙江省杭州市西湖区文三路123号5楼"}]
for adapter in [None, "adapters/logistics-lora"]:
    model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit", adapter_path=adapter)
    prompt = tok.apply_chat_template(msgs, tools=openai_tools(all_specs()), add_generation_prompt=True)
    print("便签本:", adapter, "→", generate(model, tok, prompt=prompt, max_tokens=120, verbose=False)[:150], "\n")
