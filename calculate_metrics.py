import pandas as pd
import os
import json
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import defaultdict
from typing import Dict
from utils import extract_and_parse_json


class EloCalculator:
    def __init__(
        self,
        method="mle",
        K=4,
        scale=400,
        base=10,
        init_rating=1000,
    ) -> None:
        assert method in ["online", "mle", "whr"]
        self.method = method
        self.K = K
        self.scale = scale
        self.base = base
        self.init_rating = init_rating

    def score(self, matches) -> Dict[str, float]:
        if self.method == "online":
            return self.compute_online_elo(matches)
        elif self.method == "mle":
            return self.compute_mle_elo(matches)
        elif self.method == "whr":
            return self.compute_whr(matches)
        else:
            raise NotImplementedError

    def get_bootstrap_result(self, matches, num_round=100):
        rows = []
        matches_df = pd.DataFrame(matches, columns=["model_a", "model_b", "winner"])
        for i in tqdm(range(num_round), desc="bootstrap"):
            shuffled_matches = matches_df.sample(frac=1.0, replace=True)
            matches = list(
                zip(
                    shuffled_matches["model_a"],
                    shuffled_matches["model_b"],
                    shuffled_matches["winner"],
                )
            )
            rows.append(self.score(matches))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    def compute_whr(self, matches):
        """Compute ELO via the whole-history-rating package.

        https://github.com/pfmonville/whole_history_rating
        """
        from whr import whole_history_rating

        ret = defaultdict(lambda: self.init_rating)
        whr = whole_history_rating.Base()
        for _, match in enumerate(matches):
            if match[2] != "tie":
                whr.create_game(
                    match[0], match[1], "B" if match[2] == match[0] else "W", 1, 0
                )
            else:
                whr.create_game(match[0], match[1], "B", 1, 0)
                whr.create_game(match[0], match[1], "W", 1, 0)

        whr.auto_iterate(time_limit=10, precision=1e-1, batch_size=100)
        ratings = whr.get_ordered_ratings(current=True, compact=False)
        for model, rating in ratings:
            ret[model] = self.init_rating + rating
        return ret

    def compute_mle_elo(self, matches):
        """Compute Elo ratings for a set of matches using maximum likelihood estimation.

        That means, we calculate the MLE estimator of the Bradley-Terry model for the given matches.
        """
        rating = defaultdict(lambda: self.init_rating)

        from sklearn.linear_model import LogisticRegression

        df = pd.DataFrame(matches, columns=["model_a", "model_b", "winner"])
        models = pd.concat([df["model_a"], df["model_b"]]).unique()
        models = pd.Series(np.arange(len(models)), index=models)

        # duplicate battles
        df = pd.concat([df, df], ignore_index=True)
        p = len(models.index)
        n = df.shape[0]

        X = np.zeros([n, p])
        X[np.arange(n), models[df["model_a"]]] = +math.log(self.base)
        X[np.arange(n), models[df["model_b"]]] = -math.log(self.base)

        # one A win => two A win
        Y = np.zeros(n)
        Y[df["winner"] == df["model_a"]] = 1.0

        # one tie => one A win + one B win
        # find tie + tie (both bad) index
        tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
        tie_idx[len(tie_idx) // 2 :] = False
        Y[tie_idx] = 1.0

        model = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
        model.fit(X, Y)

        elo_scores = self.scale * model.coef_[0] + self.init_rating
        for i in range(p):
            rating[models.index[i]] = elo_scores[i]
        return rating

    def compute_online_elo(self, matches):
        """Compute online Elo ratings for a set of matches.

        Elo score source: https://en.wikipedia.org/wiki/Elo_rating_system
        E_a = 1 / (1 + 10^((R_b - R_a) / 400))
        E_b = 1 / (1 + 10^((R_a - R_b) / 400))
        Args:
        matches: List of tuples of the form (model_a, model_b, winner), where model_a and model_b are the names of the models.
            Winner is the name of the winner
        """
        rating = defaultdict(lambda: self.init_rating)

        for model_a, model_b, winner in matches:
            ra = rating[model_a]
            rb = rating[model_b]
            ea = 1 / (1 + self.base ** ((rb - ra) / self.scale))
            eb = 1 / (1 + self.base ** ((ra - rb) / self.scale))
            if winner == model_a:
                sa = 1
            elif winner == model_b:
                sa = 0
            elif winner == "tie":
                sa = 0.5
            else:
                raise Exception(f"unexpected vote {winner}")
            rating[model_a] += self.K * (sa - ea)
            rating[model_b] += self.K * (1 - sa - eb)

        return rating


def win_rate_over_model(matches, eval_model_name, baseline_model_name):
    """Compute the win rate of eval_model_name over baseline_model_name."""
    wins = 0
    total = 0
    for model_a, model_b, winner in matches:
        if sorted([model_a, model_b]) == sorted([baseline_model_name, eval_model_name]):
            total += 1
            if winner == eval_model_name:
                wins += 1
            elif winner == "tie":
                wins += 0.5
    if total == 0:
        return float("nan")
    else:
        return wins / total


def plot_win_rate(win_rate, model_list, subset):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f"RPBench-{subset} win rate matrix (Y-axis over X-axis)")
    cax = ax.matshow(win_rate, cmap="RdPu")
    ax.set_xticks(np.arange(len(model_list)))
    ax.set_yticks(np.arange(len(model_list)))
    ax.set_xticklabels([ele for ele in model_list], fontsize=8)
    ax.set_yticklabels([ele for ele in model_list], fontsize=8)
    for (i, j), z in np.ndenumerate(win_rate):
        if z > 0.5:
            color = "w"
        else:
            color = "k"
        ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center", c=color)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    fig.colorbar(cax)
    fig.tight_layout()

    return fig


def get_metrics(label_result_dir, elo_algo="mle"):
    model_num_annotations = defaultdict(int)
    matches = []

    elo_calculator = EloCalculator(method=elo_algo)

    for file in os.listdir(label_result_dir):
        if file.endswith(".jsonl"):
            with open(os.path.join(label_result_dir, file), "r") as f:
                for line in f:
                    obj = json.loads(line)
                    model_assignment = obj["assignment"]
                    winner = extract_and_parse_json(obj["judger_response"])["winner"]
                    winner_model = model_assignment.get(winner)
                    if winner_model is None:
                        continue
                    loser_model = (
                        model_assignment["model_a"]
                        if winner == "model_b"
                        else model_assignment["model_b"]
                    )
                    matches.append((winner_model, loser_model, winner_model))
                    model_num_annotations[winner_model] += 1
                    model_num_annotations[loser_model] += 1

    ratings = elo_calculator.score(matches)

    model_list = sorted(model_num_annotations.keys())
    win_rate = np.zeros((len(model_list), len(model_list)))

    for i, model_a in enumerate(model_list):
        for j, model_b in enumerate(model_list):
            if i != j:
                win_rate[i, j] = win_rate_over_model(matches, model_a, model_b)
            else:
                win_rate[i, j] = float("nan")

    ratings = pd.DataFrame(
        [
            {
                "model_id": model,
                "model_name": model,
                "elo_rating": ratings[model],
                "num_annotations": model_num_annotations[model],
            }
            for model in ratings
        ]
    )
    ratings = ratings.sort_values(by="elo_rating", ascending=False)

    return ratings, win_rate, model_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_result_dir", type=str, default="results")
    parser.add_argument(
        "--subset", type=str, choices=["character", "scene"], required=True
    )
    parser.add_argument(
        "--elo_algo", type=str, choices=["online", "mle", "whr"], default="mle"
    )
    args = parser.parse_args()
    ratings, win_rate, model_list = get_metrics(
        os.path.join(args.label_result_dir, args.subset), elo_algo=args.elo_algo
    )
    print(ratings)
    plot_win_rate(win_rate, model_list, args.subset)
    plt.show()
