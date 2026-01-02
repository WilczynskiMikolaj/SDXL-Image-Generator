import argparse
from model_configs import model_descriptions
from model_loader import ModelLoader

def run_terminal():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="spcify which model to use", action="store", choices=model_descriptions.keys(), default=None)
    parser.add_argument("-l", "--loras",help="Provide loras",nargs="+", default=[], metavar="ITEM")
    args = parser.parse_args()

    if args.model :
        selected_model = model_descriptions[args.model]
        print(f"Selected model: {args.model}")
        print(f"Model description: {selected_model}")
        print(f"Loras: {args.loras}")

    model = ModelLoader(args.model, args.loras)
    model.prompt_model()


if __name__ == "__main__":
    run_terminal()  