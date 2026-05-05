from pathlib import Path
import gradio as gr
from sdxl_image_generator.pipelines.pipeline_manager import PipelineManager
from sdxl_image_generator.utils.pipeline_configs import SDXLConfig
from sdxl_image_generator.utils.utils import GenerationConfig, LoraKey, ModelDevice, ModelType, PipelineKey, PipelineRequest, PipelineType, get_all_directory_elements, choose_folder, get_directory, read_history_from_jsonl, write_prompt_history, apply_config, save_all_images
from sdxl_image_generator.utils.utils import SELECT_FOLDER_ICON_PATH, PROMPT_HISTORY_FILE, IMAGES_OUTPUT_FOLDER, default_schedulers
from sdxl_image_generator.ui.components import random_seed_btn
import random


def create_ui():
    available_models_dict = get_all_directory_elements("model_checkpoints", project_directory=True)
    available_loras_dict = get_all_directory_elements("loras", project_directory=True)
    
    scheduler_names = ["Default", "dpmpp_2m", "euler_a", "euler", "ddim", "heun"]
    
    available_models_dict["default_sdxl"] = "stabilityai/stable-diffusion-xl-base-1.0"
    
    available_models_list = ["Default (stable-diffusion-xl-base-1.0)", *[k for k in available_models_dict.keys() if k != "default_sdxl"]]
    available_loras_list = list(available_loras_dict.keys())

    pipeline_manager = PipelineManager(
        available_models=available_models_dict,
        available_loras=available_loras_dict,
        schedulers=default_schedulers
    )
    
    initial_prompt_history = read_history_from_jsonl(PROMPT_HISTORY_FILE) or []
    
    if len(initial_prompt_history) > 0:
        last_prompt_id = initial_prompt_history[-1].get("prompt_id", 0)
    else:
        last_prompt_id = 0
    
    with gr.Blocks(fill_width=True, fill_height=True) as demo:
        lora_element_state = gr.State({})
        prompt_history_state = gr.State(initial_prompt_history)
        current_prompt_id = gr.State(last_prompt_id + 1)
        images_output_dir = gr.State(IMAGES_OUTPUT_FOLDER)
    

        gr.Markdown("# SDXL GENERATOR GUI")
        with gr.Row(equal_height=True):
            with gr.Tab("Generation", scale=1):
                with gr.Group():
                    models_dropdown = gr.Dropdown(choices=available_models_list, label="Model Checkpoint", interactive=True, multiselect=False, value=available_models_list[0])
                    dropdown_button = gr.Button(value="Select Models Folder", variant="secondary", icon=SELECT_FOLDER_ICON_PATH)
                    @dropdown_button.click(outputs=[models_dropdown])
                    def get_models_path():
                        model_folder = choose_folder()
                        
                        if not model_folder:
                            return gr.update()
                            
                        models_dict = get_all_directory_elements(model_folder, project_directory=False)
                        models_dict["default_sdxl"] = "stabilityai/stable-diffusion-xl-base-1.0"
                        pipeline_manager.set_available_models(models_dict)
                        available_models = ["Default (stable-diffusion-xl-base-1.0)", *[k for k in models_dict.keys() if k != "default_sdxl"]]
                        return gr.Dropdown(choices=available_models, label="Model Checkpoint", interactive=True, multiselect=False, value=available_models[0], scale=8)
                positive_prompt = gr.Textbox(label="Positive Prompt", lines=6)
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=6)

                schedulers_dropdown = gr.Dropdown(choices=scheduler_names, label="Scheduler", interactive=True, multiselect=False, value="Default")
                width = gr.Slider(64, 6144, step=64, label="Image width", value=1024, interactive=True)
                height = gr.Slider(64, 6144, step=64, label="Image height", value=1024, interactive=True)
                inference_steps = gr.Slider(1, 100, step=1, label="Inference steps", value=30, interactive=True)
                guidance_scale = gr.Slider(0.0, 12.0, step=0.01, label="CFG", value=7.0, interactive=True)
                guidance_rescale = gr.Slider(0.0, 1.5, step=0.01, label="CFG Rescale", value=0.5, interactive=True)
                images_per_prompt = gr.Slider(1, 20, step=1, label="Images per prompt", value=1, interactive=True)
                seed = gr.Number(0, label="Seed", minimum=0, maximum=2**32 - 1, buttons=[random_seed_btn])
                random_seed_btn.click(lambda: random.randint(0, 2**32 - 1), outputs=seed)

                generate_button = gr.Button("Generate")

            with gr.Tab("Lora", scale=1):
                lora_folder_selection_button = gr.Button(value="Select Loras Folder", icon=SELECT_FOLDER_ICON_PATH)
                lora_dropdown = gr.Dropdown(choices=available_loras_list, label="Loras", interactive=True, multiselect=True)
                @lora_folder_selection_button.click(outputs=[lora_dropdown, lora_element_state])
                def get_loras_path():
                    loras_folder = choose_folder()
                    
                    if not loras_folder:
                        return gr.update(), gr.update()
                        
                    loras_dict = get_all_directory_elements(loras_folder, project_directory=False)
                    
                    pipeline_manager.set_available_loras(loras_dict)
                    
                    available_loras = list(loras_dict.keys())
                    
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
                        weight = state.get(lora, 0.0) 
                        slider = gr.Slider(0.0, 2.0, value=weight, step=0.05, label=f"{lora} weight", interactive=True)
                        slider.release(fn=lambda v, s, name=lora: update_weight(v, s, name), inputs=[slider, lora_element_state], outputs=lora_element_state)

            with gr.Tab("Configuration", scale=1):
                with gr.Group():
                    save_all_checkbox = gr.Checkbox(label="Auto save images")
                    change_output_dir_button = gr.Button("Select Image Output Folder", icon=SELECT_FOLDER_ICON_PATH)
                    
                    def safe_change_output_dir(current_dir):
                        new_folder = choose_folder()
                        return new_folder if new_folder else current_dir
                        
                    change_output_dir_button.click(
                        fn=safe_change_output_dir, 
                        inputs=[images_output_dir], outputs=images_output_dir)
                
         
            with gr.Column(scale=4):
                gallery = gr.Gallery(preview=True, object_fit="contain")

            with gr.Sidebar(position="right"):
                gr.Markdown("## Prompt History")
                @gr.render(inputs=prompt_history_state)
                def render_prompt_history(prompt_history):
                    for prompt in prompt_history:
                        title = f"Prompt {prompt['prompt_id']} -  {prompt['positive_prompt'][:20]}..."
                        with gr.Accordion(title, open=False):
                            gr.Markdown(f"""## Prompt

                            **Positive Prompt**
                            > {prompt['positive_prompt']}

                            **Negative Prompt**
                            > {prompt['negative_prompt']}
                            """)
                            load_prompt_btn = gr.Button("Load prompt")
                            load_prompt_btn.click( fn=lambda p=prompt: apply_config(p),
                                            outputs=[
                                                positive_prompt,
                                                negative_prompt,
                                                schedulers_dropdown,
                                                width,
                                                height,
                                                inference_steps,
                                                guidance_scale,
                                                guidance_rescale,
                                                images_per_prompt,
                                                seed
                                            ])

                    
        @generate_button.click(inputs=[
            models_dropdown, positive_prompt, negative_prompt, width, height, inference_steps, 
            guidance_scale, images_per_prompt, lora_element_state, seed, guidance_rescale, 
            schedulers_dropdown, prompt_history_state, current_prompt_id, save_all_checkbox, images_output_dir
        ], outputs=[gallery, prompt_history_state, current_prompt_id])
        def generate(models_dropdown, positive_prompt, negative_prompt, width, height, 
                     inference_steps, guidance_scale, images_per_prompt, lora_element_state, 
                     seed, guidance_rescale, scheduler, prompt_history, prompt_id, 
                     save_checkbox, output_dir):
            

            gen_config = GenerationConfig(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                image_width=width,
                image_height=height,
                images_per_prompt=images_per_prompt,
                seed=int(seed) if seed else None
            )

            init_loras = {}
            for lora_name, weight in lora_element_state.items():
                if weight > 0.0 and lora_name in pipeline_manager.available_loras:
                    lora_path = str(pipeline_manager.available_loras[lora_name])
                    init_loras[lora_name] = LoraKey(file_path=lora_path, adapter_weight=weight)

            model_config = SDXLConfig(
                model_checkpoint="",
                scheduler=scheduler if scheduler != "Default" else None,
                init_loras=init_loras
            )

            pipeline_key = PipelineKey(
                model_type=ModelType.SDXL,
                pipeline_type=PipelineType.TEXT2IMG,
                model_name=models_dropdown if models_dropdown != "Default (stable-diffusion-xl-base-1.0)" else "default_sdxl"
            )

            request = PipelineRequest(
                key=pipeline_key,
                config=model_config,
                device=ModelDevice.GPU,
                generation_config=gen_config
            )

            
            images = pipeline_manager.run_pipeline(request)

            
            ui_history_config = {
                "prompt_id": prompt_id, "model": models_dropdown, "positive_prompt": positive_prompt, 
                "negative_prompt": negative_prompt, "image_width": width, "image_height": height, 
                "inference_steps": inference_steps, "guidance_scale": guidance_scale, 
                "images_per_prompt": images_per_prompt, "seed": seed, "guidance_rescale": guidance_rescale, 
                "loras": lora_element_state, "scheduler": scheduler
            }

            write_prompt_history(ui_history_config, PROMPT_HISTORY_FILE)

            if save_checkbox:
                save_all_images(images, prompt_id, positive_prompt, output_dir)

            return images, [*prompt_history, ui_history_config], prompt_id + 1
            

    return demo