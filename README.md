# RPBench-Auto
[Leaderboard](https://boson.ai/rpbench/) | [Blog](https://boson.ai/rpbench-blog/)

An automated pipeline for evaluating LLMs for role-playing.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
First, set the environment variable `OPENAI_API_KEY` for the judge model and  to the path of the RPBench dataset.
```bash
export OPENAI_API_KEY=<API_KEY>
```

Then, add the model config file for the model you want to evaluate. Currently we support OpenAI API (and compatible APIs) and Anthropic API. Edit [config/api_config.yaml](config/api_config.yaml) to add the model config.

Finally, run the pipeline.
```bash
python run_character_eval.py --model_1 <CONFIG_NAME>  # Evaluate the model on the character subset
python run_scene_eval.py --model_1 <CONFIG_NAME>  # Evaluate the model on the scene subset
```

Generate the leaderboard.
```bash
python generate_leaderboard.py
```

## How to contribute
After running all commands above, you can add your model to the leaderboard by creating a pull request with the updated leaderboard files, `leaderboard.csv` and `leaderboard_for_display.csv`, plus the .jsonl files in `/results/character` and `/results/scene`. The leaderboard will be updated automatically when the PR is merged.

## Acknowledgements
This benchmark is heavily inspired by [ArenaHard](https://github.com/lm-sys/arena-hard-auto) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/). Some code implementations are borrowed from these repositories.
