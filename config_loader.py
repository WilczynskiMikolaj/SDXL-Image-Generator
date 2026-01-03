import json

def load_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = {}

    if "model_name" in data:
        config["model_name"] = data["model_name"]

    if "loras" in data:
        config["loras"] = data["loras"]

    if "image_size" in data:
        config["image_size"] = tuple(data["image_size"])

    if "inference_steps" in data:
        config["inference_steps"] = data["inference_steps"]

    if "guidance_scale" in data:
        config["guidance_scale"] = data["guidance_scale"]

    if "images_per_prompt" in data:
        config["images_per_prompt"] = data["images_per_prompt"]

    if "adapter_weights" in data:
        config["adapter_weights"] = data["adapter_weights"]

    return config