import os

def get_lora_names(folder="loras", output_file="lora_names.txt"):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found.")
        return []

    # Get only file names (no directories)
    lora_files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ]

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for name in lora_files:
            f.write(name + "\n")

    print(f"Found {len(lora_files)} files. Saved to {output_file}")
    return lora_files

get_lora_names()