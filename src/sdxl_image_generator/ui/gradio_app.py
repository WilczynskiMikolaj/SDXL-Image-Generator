import gradio as gr
from sdxl_image_generator.utils.utils import get_all_directory_elements


def create_ui():
    model_checkpoint_names = get_all_directory_elements("model_checkpoints")
    lora_names = get_all_directory_elements("loras")
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:
        lora_element_state = gr.State([{"name":lora, "enabled": False} for lora in lora_names])

        gr.Markdown("# SDXL-CLI-GENERATOR GUI")
        with gr.Row(equal_height=True):
            with gr.Tab("Generation", scale=1):

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
                @gr.render(inputs=lora_element_state)
                def render_lora_selection(lora_list):
                    for lora in lora_list:
                        lora_checkbox = gr.Checkbox(interactive=True, label=lora["name"])
                        gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Adapter Weight", interactive=lora["enabled"])
                        def select_lora(lora=lora):
                            if lora["enabled"] == True:
                                lora["enabled"] = False
                            else:
                                lora["enabled"] = True
                            return lora_list
                        lora_checkbox.select(select_lora, None, lora_element_state)
                        
                    
                    
            with gr.Column(scale=4):
                gallery = gr.Gallery(preview=True, object_fit="contain")

        @generate_button.click(inputs=[models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, guidance_scale, images_per_prompt, lora_element_state], outputs=gallery)
        def generate():
            return None

    return demo