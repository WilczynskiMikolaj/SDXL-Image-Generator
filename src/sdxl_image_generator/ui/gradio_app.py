import gradio as gr
from sdxl_image_generator.utils.utils import get_all_directory_elements


def create_ui():
    model_checkpoint_names = get_all_directory_elements("model_checkpoints")
    lora_names = get_all_directory_elements("loras")
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:

        gr.Markdown("# SDXL-CLI-GENERATOR GUI")
        with gr.Row(equal_height=True, scale=1):
            with gr.Tab("Generation"):

                models_dropdown = gr.Dropdown(choices=model_checkpoint_names, label="Model Checkpoint", interactive=True)
                positive_prompt = gr.Textbox(label="Positive Prompt", lines=6)
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=6)

                width = gr.Slider(64, 6144, step=64, label="Image width", value=1024, interactive=True)
                height = gr.Slider(64, 6144, step=64, label="Image height", value=1024, interactive=True)
                inference_steps = gr.Slider(1, 100, step=1, label="Inference steps", value=30, interactive=True)
                guidance_scale = gr.Slider(3.0, 12.0, step=0.01, label="Guidance scale", value=7.0, interactive=True)
                images_per_prompt = gr.Slider(1, 20, step=1, label="Images per prompt", value=1, interactive=True)

                generate_button = gr.Button("Generate")

            with gr.Tab("Lora", scale=1):
                models_dropdown = gr.Dropdown(label="Model Checkpoint")
            with gr.Column(scale=4):
                gallery = gr.Gallery(preview=True, object_fit="contain")

    return demo