import gradio as gr
from sdxl_image_generator.utils.utils import get_all_directory_elements, choose_folder, get_directory
from sdxl_image_generator.utils.utils import ICON_PATH
from sdxl_image_generator.sdxl_model_pipeline.model_loader_for_ui import ModelLoaderUI

def create_ui():
    model_checkpoint_names = get_all_directory_elements("model_checkpoints", project_directory=True)
    lora_names = get_all_directory_elements("loras", project_directory=True)
    scheduler_names = ["Default", "dpmpp_2m", "euler_a", "euler", "ddim", "heun"]
    available_models = ["Default (stable-diffusion-xl-base-1.0)", *model_checkpoint_names]
    available_loras = [*lora_names]

    model_loader = ModelLoaderUI(available_models, lora_names)
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:
        lora_element_state = gr.State({})
        chat_history = []

        gr.Markdown("# SDXL GENERATOR GUI")
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
                lora_folder_selection_button = gr.Button(value="Select Loras Folder", icon=ICON_PATH)
                lora_dropdown = gr.Dropdown(choices=available_loras, label="Loras", interactive=True, multiselect=True)
                @lora_folder_selection_button.click(outputs=[lora_dropdown, lora_element_state])
                def get_models_path():
                    loras_folder = choose_folder()
                    available_loras = [*get_all_directory_elements(loras_folder, project_directory=False)]
                    model_loader.load_loras_selection(available_loras, get_directory(loras_folder))
                    state_dict = {}
                    for lora in available_loras:
                        state_dict.update({lora: 0.0})
                    return gr.Dropdown(choices=available_loras, value=None, label="Loras", interactive=True, multiselect=True, scale=8), state_dict

                def update_weight(value, state, lora_name):
                    state[lora_name] = value
                    return state
                
                @gr.render(inputs=[lora_dropdown, lora_element_state])
                def render_lora_sliders(selected_loras, state):
                    selected_loras = selected_loras or []
                    for lora in selected_loras:
                        weight = state[lora]
                        slider = gr.Slider(0.0, 2.0, value=weight, step=0.05, label=f"{lora} weight", interactive=True)
                        slider.release(fn=lambda v, s, name=lora: update_weight(v, s, name), inputs=[slider, lora_element_state], outputs=lora_element_state)
         
            with gr.Column(scale=4):
                gallery = gr.Gallery(preview=True, object_fit="contain")

            with gr.Sidebar(position="right"):
                prompt_history = gr.Chatbot(chat_history)

        @generate_button.click(inputs=[models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, guidance_scale, images_per_prompt, lora_element_state, seed, guidance_rescale, schedulers_dropdown, prompt_history], outputs=[gallery, prompt_history])
        def generate(models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, guidance_scale, images_per_prompt, lora_element_state, seed, guidance_rescale, scheduler, prompt_history):
            config = {"model": models_dropdown, "prompt": positive_prompt, "negative_prompt": negative_prompt, "image_width": width, "image_height": height, "inference_steps": inference_steps, "guidance_scale": guidance_scale, 
                      "images_per_prompt": images_per_prompt, "seed": seed, "guidance_rescale": guidance_rescale, "loras": lora_element_state, "scheduler": scheduler}

            model_loader.load_model(model_name=config["model"])

            loras_selected = list(lora_element_state.keys())
            adapter_weights = list(lora_element_state.values())
            model_loader.load_loras(loras=loras_selected, adapter_weights=adapter_weights)

            model_loader.initialize_compel()
            images = model_loader.generate_images(config=config)
            prompt_history.append({"role": "user", "content": positive_prompt})
            return images, prompt_history
            

    return demo