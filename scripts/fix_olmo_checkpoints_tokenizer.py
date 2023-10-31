"""a script to switch olmo checkpoint config.yamls to use the hf tokenizer"""

import argparse
import os
import yaml

def main(args):
    for checkpoint_dir in os.listdir(args.checkpoints_dir):
        config_path = os.path.join(args.checkpoints_dir, checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            assert config["tokenizer"]["identifier"] == "tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json", "all configs should have bad problem tokenizer before fix"
            config["tokenizer"]['identifier'] = "allenai/eleuther-ai-gpt-neox-20b-pii-special"

            # overwrite config.yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)