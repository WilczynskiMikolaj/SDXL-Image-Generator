import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown( """ # SDXL-CLI-GENERATOR GUI """)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            models_dropdown = gr.Dropdown()
            positive_prompt = gr.Textbox(label="Positive Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")


demo.launch()