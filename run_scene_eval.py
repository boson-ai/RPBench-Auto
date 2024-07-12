import os
import json
import jsonlines
from utils import make_config, chat_completion, extract_and_parse_json
from string import Template
from tqdm.auto import tqdm
import random
import argparse

MAX_MESSAGES_PER_CHAR = 10
RPBENCH_PATH = "data/rpbench_scene.jsonl"


TEMPLATE = Template(
    """# Background:    
$background

# NPC Profile:
$npc_profile

# Previous plots recap:
$plot_recap

# Current Objectives:
$objectives

# When to end the scene:
$progression

# NPC Status:
$npc_status

You are an AI NPC for a role-play game. Based on the previous information, you will play a NPC character in the scene.
Reply in this JSON format: {"npc_speaks": "YOUR RESPONSE", "is_chat_finished": true or false}. If the scene is finished and ready to move on, set "is_chat_finished" to true. Otherwise, set it to false.
"""
)

JUDGER_TEMPLATE = Template(
    """# Background:    
$background

# NPC Profile:
$npc_profile

# Previous plots recap:
$plot_recap

# Current Objectives:
$objectives

# When to end the scene:
$progression

# NPC Status:
$npc_status

You are testing a game powered by AI NPCs. You need to simulate a player and interact with 2 AI NPC. The NPC will responds with its reply and determine if the scene meets the criteria to end and move on to the next plot. For each round (except the first round), you should pick a better response from the 2 AI NPC and come up with your reply. It will be in a JSON format: {"winner": "model_a" or "model_b", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "REASON FOR PICK THE WINNER"}. For the first round, use "winner": null. If the winner you pick have "is_chat_finished" set to true, you can leave "next_round_user_speaks" empty.
"""
)


def chat_completion_judger(model, messages):
    while True:
        response = chat_completion(model, messages)
        try:
            parsed_response = extract_and_parse_json(response)
            if (
                "winner" in parsed_response
                and "next_round_user_speaks" in parsed_response
            ):
                return response
        except:
            pass


def eval_models_pairwise(model_1, model_2):
    model_1_win_count = 0
    model_2_win_count = 0
    eval_data = []
    win_lose_pairs = []
    eval_results = []
    with jsonlines.open(RPBENCH_PATH) as reader:
        for obj in reader:
            eval_data.append(obj)
    print(f"Loaded {len(eval_data)} examples from {RPBENCH_PATH}")

    judger_config = make_config("config/judger_config.yaml")
    assert len(judger_config) == 1, "Judger config should have only one model"
    judger_model_name = list(judger_config.keys())[0]
    judger_model = judger_config[judger_model_name]
    print(f"Judger model: `{judger_model_name}`")

    candidate_config = make_config("config/api_config.yaml")
    assert model_1 in candidate_config, f"{model_1} not found in candidate config"
    assert model_2 in candidate_config, f"{model_2} not found in candidate config"
    print(f"Comparing `{model_1}` and `{model_2}`")

    for d in (pbar := tqdm(eval_data)):
        conversation = d["conversation"]
        greeting = "\n".join(conversation[0]["sentences"])
        candidate_messages = [
            {
                "role": "system",
                "content": TEMPLATE.substitute(d),
            },
            {"role": "assistant", "content": greeting},
        ]

        judger_messages = [
            {"role": "system", "content": JUDGER_TEMPLATE.substitute(d)},
            {
                "role": "user",
                "content": json.dumps({"model_a": greeting, "model_b": greeting}),
            },
        ]

        judger_response = chat_completion_judger(judger_model, judger_messages)
        parsed_judger_response = extract_and_parse_json(judger_response)
        judger_messages.append({"role": "assistant", "content": judger_response})

        for _ in range(MAX_MESSAGES_PER_CHAR):
            # randomly assign model_a and model_b to model_1 and model_2
            model_a = model_1 if bool(random.getrandbits(1)) else model_2
            model_b = model_2 if model_a == model_1 else model_1
            assignment = {"model_a": model_a, "model_b": model_b}

            user_input = parsed_judger_response["next_round_user_speaks"]
            candidate_messages.append({"role": "user", "content": user_input})
            model_a_response = chat_completion(
                candidate_config[model_a], candidate_messages
            )
            model_b_response = chat_completion(
                candidate_config[model_b], candidate_messages
            )

            try:
                parsed_model_a_response = extract_and_parse_json(model_a_response)
                model_a_end = parsed_model_a_response["is_chat_finished"]
            except:
                model_a_end = False
                print(f"Warning: Format error in response of {model_a}")
                print(model_a_response)
            try:
                parsed_model_b_response = extract_and_parse_json(model_b_response)
                model_b_end = parsed_model_b_response["is_chat_finished"]
            except:
                model_b_end = False
                print(f"Warning: Format error in response of {model_b}")
                print(model_b_response)

            judger_message_content = json.dumps(
                {"model_a": model_a_response, "model_b": model_b_response}
            )
            judger_messages.append({"role": "user", "content": judger_message_content})
            judger_response = chat_completion_judger(judger_model, judger_messages)
            parsed_judger_response = extract_and_parse_json(judger_response)

            eval_result = {
                "candidate_messages": candidate_messages,
                "assignment": assignment,
                "judger_messages": judger_messages,
                "judger_response": judger_response,
            }
            eval_results.append(eval_result)
            winner = parsed_judger_response["winner"]
            if winner:
                winner_model = None
                if winner == "model_a":
                    winner_model = model_a
                    win_lose_pairs.append((model_a, model_b))
                elif winner == "model_b":
                    winner_model = model_b
                    win_lose_pairs.append((model_b, model_a))
                if winner_model == model_1:
                    model_1_win_count += 1
                elif winner_model == model_2:
                    model_2_win_count += 1

                if winner == "model_a" and model_a_end:
                    break
                elif winner == "model_b" and model_b_end:
                    break

            pbar.set_postfix(
                {
                    "model_1_win_rate": model_1_win_count
                    / (model_1_win_count + model_2_win_count)
                }
            )

            judger_messages.append({"role": "assistant", "content": judger_response})
            candidate_messages.append(
                {
                    "role": "assistant",
                    "content": model_a_response
                    if winner == "model_a"
                    else model_b_response,
                }
            )

    if not os.path.exists("results/scene"):
        os.makedirs("results/scene")
    with jsonlines.open(
        f"results/scene/eval_{model_1}_vs_{model_2}.jsonl", "w"
    ) as writer:
        writer.write_all(eval_results)

    return win_lose_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, default="gpt-4o")
    args = parser.parse_args()
    eval_models_pairwise(args.model_1, args.model_2)
