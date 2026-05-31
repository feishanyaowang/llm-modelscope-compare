import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

models = {
    "DeepSeek-R1-Distill-Qwen-7B": "/mnt/data/models/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct": "/mnt/data/models/Qwen2.5-7B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "/mnt/data/models/Qwen2.5-Coder-7B-Instruct",
}

questions = [
    "请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思，小意思。领导：你这人真有意思。小明：其实也没有别的意思。领导：那我就不好意思了。小明：是我不好意思。请问以上每个“意思”分别是什么意思？"
]

os.makedirs("/mnt/workspace/results_7b", exist_ok=True)

for model_name, model_path in models.items():
    print("\n" + "=" * 80)
    print("加载模型:", model_name)
    print("模型路径:", model_path)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    result_path = f"/mnt/workspace/results_7b/{model_name}.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"模型：{model_name}\n")
        f.write(f"路径：{model_path}\n")
        f.write("=" * 80 + "\n\n")

        for idx, question in enumerate(questions, 1):
            print(f"\n[{model_name}] 测试问题 {idx}")

            messages = [
                {
                    "role": "user",
                    "content": "请用中文直接回答，尽量简洁，不要输出思考过程。\n" + question
                }
            ]

            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = messages[0]["content"]

            inputs = tokenizer(
                text,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            answer = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()

            print(answer)

            f.write(f"问题 {idx}：\n")
            f.write(question + "\n\n")
            f.write("回答：\n")
            f.write(answer + "\n\n")
            f.write("-" * 80 + "\n\n")

    print("结果已保存:", result_path)

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n全部 7B 模型测试完成，结果保存在 /mnt/workspace/results_7b")
