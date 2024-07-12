import os
import argparse
import yaml
from calculate_metrics import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_result_dir", type=str, default="results")
    parser.add_argument("--model_config", type=str, default="./config/api_config.yaml")
    parser.add_argument("--baseline_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    result_dfs = []

    for subset in ["character", "scene"]:
        _, win_rate, model_list = get_metrics(
            os.path.join(args.label_result_dir, subset), elo_algo="online"
        )
        # get model_name -> beautiful name mapping
        # read yaml
        with open(args.model_config, "r") as f:
            model_config = yaml.load(f, Loader=yaml.SafeLoader)
        model_name_to_beautiful_name = {}
        for k, v in model_config.items():
            model_name_to_beautiful_name[k] = v["beautiful_name"]
        
        baseline_model_index = model_list.index(args.baseline_model)

        for i in range(len(model_list)):
            win_rate[i, i] = 0.50
        
        win_rate_df = pd.DataFrame(
            {
                "model": model_list,
                "beautiful_name": [model_name_to_beautiful_name[model] for model in model_list],
                "win_rate": win_rate[:, baseline_model_index],
            }
        )

        # rank by win rate
        win_rate_df = win_rate_df.sort_values(by="win_rate", ascending=False)
        result_dfs.append(win_rate_df)

    result_dfs[1] = result_dfs[1].drop(columns=["beautiful_name"])
    # merge the two dataframes by field "model"
    leaderboard_df = pd.merge(result_dfs[0], result_dfs[1], on="model", suffixes=("_character", "_scene"))
    # add win_rate_avg
    leaderboard_df["win_rate_avg"] = (leaderboard_df["win_rate_character"] + leaderboard_df["win_rate_scene"]) / 2
    # rank by win_rate_avg
    leaderboard_df = leaderboard_df.sort_values(by="win_rate_avg", ascending=False)
    # reindex, start from 1
    leaderboard_df.index = np.arange(1, len(leaderboard_df) + 1)
    leaderboard_df.index.name = "rank"
    print(leaderboard_df)
    # set index to model and sort it by model
    leaderboard_df = leaderboard_df.set_index("model")
    leaderboard_df = leaderboard_df.sort_index()
    # save to csv
    leaderboard_df.to_csv("./results/leaderboard.csv")
    # drop model column
    leaderboard_df = leaderboard_df.reset_index()
    leaderboard_df = leaderboard_df.drop(columns=["model"])
    # rank by win_rate_avg
    leaderboard_df = leaderboard_df.sort_values(by="win_rate_avg", ascending=False)
    # rename beautiful_name to "Model"
    leaderboard_df = leaderboard_df.rename(columns={"beautiful_name": "Model", "win_rate_avg": "Avg. Win Rate", "win_rate_character": "Character", "win_rate_scene": "Scene"})
    # add a rank column
    leaderboard_df["Rank"] = np.arange(1, len(leaderboard_df) + 1)
    # reorder columns
    leaderboard_df = leaderboard_df[["Rank", "Model", "Character", "Scene", "Avg. Win Rate"]]
    # win rate to percentage
    leaderboard_df["Character"] = leaderboard_df["Character"].apply(lambda x: f"{x:.2%}")
    leaderboard_df["Scene"] = leaderboard_df["Scene"].apply(lambda x: f"{x:.2%}")
    leaderboard_df["Avg. Win Rate"] = leaderboard_df["Avg. Win Rate"].apply(lambda x: f"{x:.2%}")
    # save to csv
    leaderboard_df.to_csv("./results/leaderboard_for_display.csv", index=False)
