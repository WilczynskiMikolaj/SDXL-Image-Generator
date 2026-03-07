import gradio as gr
from sdxl_image_generator.utils.utils import get_all_directory_elements, choose_folder, get_directory
from sdxl_image_generator.utils.utils import ICON_PATH
from sdxl_image_generator.sdxl_model_pipeline.model_loader_for_ui import ModelLoaderUI

def create_ui():
    model_checkpoint_names = get_all_directory_elements("model_checkpoints", project_directory=True)
    lora_names = get_all_directory_elements("loras", project_directory=True)
    scheduler_names = ["Default", "dpmpp_2m", "euler_a", "euler", "ddim", "heun"]

    model_loader = ModelLoaderUI(model_checkpoint_names, lora_names)
    available_models = ["Default", *model_checkpoint_names]
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:
        lora_element_state = gr.State([{"name":lora, "enabled": False, "adapter_weight": 0.0} for lora in lora_names])

        gr.Markdown("# SDXL-CLI-GENERATOR GUI")
        with gr.Row(equal_height=True):
            with gr.Tab("Generation", scale=1):
                with gr.Group():
                    models_dropdown = gr.Dropdown(choices=available_models, label="Model Checkpoint", interactive=True, multiselect=False, value=available_models[0])
                    dropdown_button = gr.Button(value="Select Models Folder", variant="secondary", icon=ICON_PATH)
                    @dropdown_button.click(outputs=[models_dropdown])
                    def get_models_path():
                        model_folder = choose_folder()
                        available_models = ["Default", *get_all_directory_elements(model_folder, project_directory=False)]
                        model_loader.load_model_selection(available_models, get_directory(model_folder))
                        return gr.Dropdown(choices=available_models, label="Model Checkpoint", interactive=True, multiselect=False, value=available_models[0], scale=8)
                schedulers_dropdown = gr.Dropdown(choices=scheduler_names, label="Scheduler", interactive=True, multiselect=False, value="Default")
                positive_prompt = gr.Textbox(label="Positive Prompt", lines=6)
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=6)

                width = gr.Slider(64, 6144, step=64, label="Image width", value=1024, interactive=True)
                height = gr.Slider(64, 6144, step=64, label="Image height", value=1024, interactive=True)
                inference_steps = gr.Slider(1, 100, step=1, label="Inference steps", value=30, interactive=True)
                guidance_scale = gr.Slider(0.0, 12.0, step=0.01, label="CFG", value=7.0, interactive=True)
                guidance_rescale = gr.Slider(0.0, 1.5, step=0.01, label="CFG Rescale", value=0.5, interactive=True)
                images_per_prompt = gr.Slider(1, 20, step=1, label="Images per prompt", value=1, interactive=True)
                seed = gr.Number(0, label="Seed", minimum=0, maximum=2147483647)

                generate_button = gr.Button("Generate")

            with gr.Tab("Lora", scale=1):

                lora_checkboxes = []
                lora_weight_sliders = []

                for i, lora_name in enumerate(lora_names):
                    lora_checkbox = gr.Checkbox(label=lora_name, value=False)
                    lora_slider = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Adapter Weight", interactive=False)

                    lora_checkboxes.append(lora_checkbox)
                    lora_weight_sliders.append(lora_slider)

                    def on_check(value, idx=i):
                        state = lora_element_state.value
                        state[idx]["enabled"] = value
                        return state

                    def on_weight(value, idx=i):
                        state = lora_element_state.value
                        state[idx]["adapter_weight"] = value
                        return state

                    lora_checkbox.change(on_check, lora_checkbox, lora_element_state)
                    lora_slider.change(on_weight, lora_slider, lora_element_state)

                    def toggle_slider(checked):
                        return gr.update(interactive=checked)

                    lora_checkbox.change(toggle_slider, lora_checkbox, lora_slider)

                                
            with gr.Column(scale=4):
                gallery = gr.Gallery(preview=True, object_fit="contain")

        @generate_button.click(inputs=[models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, guidance_scale, images_per_prompt, lora_element_state, seed, guidance_rescale, schedulers_dropdown], outputs=gallery)
        def generate(models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, guidance_scale, images_per_prompt, lora_element_state, seed, guidance_rescale, schedulers_dropdown):
            config = {"model": models_dropdown, "prompt": positive_prompt, "negative_prompt": negative_prompt, "image_width": width, "image_height": height, "inference_steps": inference_steps, "guidance_scale": guidance_scale, 
                      "images_per_prompt": images_per_prompt, "seed": seed, "guidance_rescale": guidance_rescale}

            model_loader.load_model(model_name=config["model"])

            loras_selected = [lora["name"] for lora in lora_element_state if lora["enabled"]]
            adapter_weights = [lora["adapter_weight"] for lora in lora_element_state if lora["enabled"]]
            model_loader.load_loras(loras=loras_selected, adapter_weights=adapter_weights)

            model_loader.initialize_compel()
            images = model_loader.generate_images(config=config)
            return images
            

    return demo