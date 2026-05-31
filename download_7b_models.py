from modelscope import snapshot_download

models = {
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

for name, model_id in models.items():
    print("=" * 80)
    print("正在下载:", name)
    print("ModelScope ID:", model_id)

    local_dir = f"/mnt/data/models/{name}"

    snapshot_download(
        model_id,
        local_dir=local_dir
    )

    print("下载完成:", local_dir)
