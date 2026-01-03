# SDXL-CLI-Generator

A simple Stable Diffusion XL (SDXL) pipeline that allows generating images directly from the command line using custom prompts, models, and LoRAs.

The main purpose of this project is to experiment with trained SDXL checkpoints and LoRAs and generate fun images. It was a challenging and enjoyable personal project to build.

---

## 🚀 How to Use

### 1️⃣ Write Your Prompt
Edit **`prompt.json`** and fill in:
- `prompt`
- `negativePrompt` (optional)

### 2️⃣ (Optional) Add Model Checkpoints
Place your `.safetensors` SDXL checkpoint files into:

### 3️⃣ (Optional) Add LoRAs
Place your `.safetensors` LoRA files into:

### 4️⃣ Run the Generator
Run the script:
main.py

### ✔️ Optional Arguments
Use a specific model:
-m flag
Use one or more LoRAs:
-l flag

## 🧭 Plans
- Add loading the model configuration with json file ✅
- Add more terminal options ✅
- terminal options overwrite json defaults ✅
- Add a GUI

---

