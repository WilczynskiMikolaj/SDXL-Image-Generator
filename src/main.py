import argparse
from src.sdxl_model_pipeline.model_configs import model_descriptions
from src.sdxl_model_pipeline import ModelLoader

def run_terminal():
    parser = argparse.ArgumentParser(
        prog="model_runner",
        description="Run an image generation model with optional JSON config and CLI overrides.",
        epilog="CLI arguments override JSON values, which override built-in defaults.",
    )

    parser.add_argument(
        "-j", "--json",
        metavar="FILE",
        help="Path to a JSON configuration file (used as defaults).",
    )

    parser.add_argument(
        "-m", "--model",
        choices=model_descriptions.keys(),
        help="Model identifier to use.",
    )

    parser.add_argument(
        "-l", "--loras",
        nargs="+",
        metavar="LORA",
        help="One or more LoRA adapters to apply (space-separated).",
    )

    parser.add_argument(
        "-W", "--width",
        type=int,
        metavar="PX",
        help="Output image width in pixels (default: 1024).",
    )

    parser.add_argument(
        "-H", "--height",
        type=int,
        metavar="PX",
        help="Output image height in pixels (default: 1024).",
    )

    parser.add_argument(
        "-g", "--guidance_scale",
        type=float,
        metavar="FLOAT",
        help="Classifier-free guidance scale (e.g. 3.5–8.0).",
    )

    parser.add_argument(
        "-i", "--inference_steps",
        type=int,
        metavar="N",
        help="Number of inference steps.",
    )

    parser.add_argument(
        "-a", "--adapter_weights",
        nargs="+",
        type=float,
        metavar="W",
        help="Adapter weights for each LoRA (must match number of --loras).",
    )

    parser.add_argument(
        "-q", "--image_quantity",
        type=int,
        metavar="N",
        help="Number of images to generate per prompt.",
    )

    args = parser.parse_args()

    model = ModelLoader.from_cli(args)
    model.prompt_model()

if __name__ == "__main__":
    run_terminal()  