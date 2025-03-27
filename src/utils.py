import json
import yaml
from openai import OpenAI
import types
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple, ClassVar
import warnings
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict, fields
import random
import time

MODEL = "gpt-4o-mini"

def retry_with_exponential_backoff(
    func, retries=20, intial_sleep_time: int = 3, jitter: bool = True, backoff_factor: float = 1.5
):
    """
    This is a sneaky function that gets around the "rate limit error" from GPT (GPT has a maximum tokens per min processed, and concurrent processing may exceed this) by retrying the model call that exceeds the limit after a certain amount of time.
    """

    def wrapper(*args, **kwargs):
        sleep_time = intial_sleep_time
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    sleep_time *= backoff_factor * (1 + jitter * random.random())
                    time.sleep(sleep_time)
                else:
                    raise
        raise Exception(f"Maximum retries {retries} exceeded")

    return wrapper


def apply_system_format(content: str) -> dict:
    return {"role": "system", "content": content}


def apply_user_format(content: str) -> dict:
    return {"role": "user", "content": content}


def apply_assistant_format(content: str) -> dict:
    return {"role": "assistant", "content": content}


def apply_message_format(user: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


@retry_with_exponential_backoff
def generate_formatted_response(client: OpenAI,
                                model = MODEL,
                                messages:Optional[List[dict]]=None,
                                user:Optional[str]=None,
                                system:Optional[str]=None,
                                temperature:int=1,
                                verbose: bool = False,
                                response_format:BaseModel=None) -> Dict[str, Any]:
    '''
    Generate a formatted response using the OpenAI API `client.beta.chat.completions.parse()` function.

    Args:
        client (OpenAI): OpenAI API client.
        messages (Optional[List[dict]], optional): List of formatted message dictionaries with 'role' and 'content' keys.
        user (Optional[str], optional): User message string (alternative to `messages`).
        system (Optional[str], optional): System message string (alternative to `messages`).
        verbose (bool, optional): If True, prints detailed message information. Defaults to False.

    Returns:
        dict: The model response as a dictionary that contains a "rt_prompts" key, which stores the list of generated red teaming prompts.

    Note:
        - If both `messages` and `user`/`system` are provided, `messages` takes precedence.

    '''
    if model != "gpt-4o-mini":
        warnings.warn(f"Warning: The model '{model}' is not 'gpt-4o-mini'.")
    
    if messages is None:
        messages = apply_message_format(user=user, system=system)

    if verbose:
        for message in messages:
            print(f"{message['role'].upper()}:\n{message['content']}\n")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=response_format,
    )

    response = completion.choices[0].message
    if response.parsed:
        return response.content

    return response

@dataclass
class GenPrompts():
    system_prompt: str
    user_prompt: str
    few_shot_examples: Optional[List[Dict]] = None
    variance_prompts: Optional[List[str]] = None

    # ================== Helper function ==================
    def get_message(self, principle, num_to_gen=1) -> List[dict]:
        """Format the system and user prompt into API message format. Return a list of system and user messages."""
        print(f"Debug: {principle}: {num_to_gen}")
        system_prompt = apply_system_format(self.system_prompt)
        
        user_prompt = apply_user_format(self.user_prompt.format(num_rtp_per_call=num_to_gen, principle=principle))
        return [system_prompt, user_prompt]


# Read principles from a json file
def get_principles_from_constitution(filename, display=False):
    with open(filename) as f:
        constitution = json.load(f)
        
        if display:
            print(f"{len(constitution)} principles found")
            print("List of principles:")
            for row in list(constitution):
                print(f"{row['title']}:{row['description']}")

        principles = [f"{row['title']}:{row['description']}" for row in list(constitution)]

    return principles

def get_config(filename, cfg_type='red_team'):
    # Load the YAML configuration file
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
        
    return config

# read red team prompts from a json file
def get_red_team_prompts(filename):
    
    with open(filename) as f:
        red_team_prompts = json.load(f)
    return red_team_prompts

if __name__ == "__main__":
    principles = get_principles_from_constitution("src/datasets/buddhist_principles.json")
    print(principles)