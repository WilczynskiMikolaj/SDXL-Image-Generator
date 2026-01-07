# SDXL Image Generator

A work-in-progress **SDXL image generation tool** with both a **CLI** and a **Gradio-based GUI**.

This project started as an experiment to build **custom SDXL pipelines** for running different checkpoints and configurations.  
Initially, I wanted only terminal usage, but during development I discovered **Gradio**, which made me want to build a GUI.

⚠️ **Project status:**  
The GUI is currently **under construction**, so CLI stoped working

---

## 🔧 Installation

### 1️⃣ Install PyTorch

Install PyTorch according to your system and CUDA version:

👉 https://pytorch.org/get-started/locally/

### 2️⃣ Install Project Requirements

All remaining dependencies are listed in `requirements.txt`.

# 🧭 Project Roadmap

## Phase 1: Configuration & CLI
- [x] Add loading the model configuration from a JSON file
- [x] Add more terminal options
- [x] Allow terminal options to overwrite JSON defaults

## Phase 2: GUI Foundation
- [x] Create basic UI layout and styling
- [x] Define and specify parameter input elements
- [x] Add dropdown options for configurable parameters

## Phase 3: Advanced UI Features
- [x] Create **LoRAs** tab
  - [ ] UI layout for LoRAs
  - [ ] LoRA selection controls
  - [ ] Enable/disable LoRAs

## Phase 4: Integration
- [ ] Connect UI to `Model_Loader`
  - [ ] Bind UI parameters to loader inputs
  - [ ] Sync JSON / CLI / UI configuration flow
  - [ ] Validate and apply runtime changes

## Phase 5: Polish & Stability

---