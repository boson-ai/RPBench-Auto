"""
Adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/utils.py
"""
import os
import json
import time
import yaml
import random
import requests
import json_repair
import re

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


def extract_and_parse_json(text):
    pattern = r"```json\s+(.+?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text

    try:
        parsed_obj = json_repair.loads(json_str)
        assert "winner" in parsed_obj
    except Exception:
        try:
            # There are something wrong in the JSON string, we will try to extract the "winner" field from the string and throw away other keys.
            winner_start = json_str.find("winner\":")
            if winner_start == -1:
                raise Exception(f"Cannot find the 'winner' field in the JSON string.\n\n{json_str}")
            winner_end = json_str.find(",", winner_start)
            new_json_str = "{\"" + json_str[winner_start:winner_end] + "}"
            parsed_obj = json_repair.loads(new_json_str)
        except Exception:
            raise Exception(f"Cannot parse JSON string.\n\nnew version={new_json_str},\n\nprevious version={json_str}")

    return parsed_obj


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def fix_anthropic_message(messages):
    # anthropic API requires the first message to be a user message
    # insert a dummy user message if the first message is a system message
    if messages[1]["role"] != "user":
        messages.insert(1, {"role": "user", "content": "Let's chat!"})
    return messages


def chat_completion(model, messages, temperature=1.0, max_tokens=2048):
    api_type = model["api_type"]
    api_dict = model.get("endpoints")
    if api_type == "anthropic":
        messages = fix_anthropic_message(messages)
        output = chat_completion_anthropic(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
    elif api_type == "mistral":
        output = chat_completion_mistral(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif api_type == "gemini":
        raise NotImplementedError(
            "Gemini API is not supported in this version due to multi-turn chat."
        )
    elif api_type == "azure":
        output = chat_completion_openai_azure(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
    elif api_type == "cohere":
        output = chat_completion_cohere(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        output = chat_completion_openai(
            model=model["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )

    return output


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    import openai

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict.get("api_base"),
            api_key=api_dict.get("api_key"),
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except TypeError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_openai_azure(
    model, messages, temperature, max_tokens, api_dict=None
):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2,
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
    ]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": message}]}],
                "safetySettings": safety_settings,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system": "SYSTEM", "assistant": "CHATBOT", "user": "USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append(
                {"role": template_map[message["role"]], "message": message["content"]}
            )
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break

    return output
