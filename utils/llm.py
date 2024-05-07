import requests
from prompt import templates, stop, required_lines, required_lines_ast
from easydict import EasyDict
from utils.cache import get_cache, add_cache
import ast
import traceback
import time
import pyjson5

model_names = [
    "vicuna",
    "vicuna-13b",
    "vicuna-13b-v1.3",
    "vicuna-33b-v1.3",
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Llama-2-70b-hf",
    "FreeWilly2",
    "gpt-3.5-turbo",
    "gpt-3.5",
    "gpt-4",
    "gpt-4-1106-preview",
]


def get_full_chat_prompt(template, prompt, suffix=None, query_prefix="Caption: "):
    # query_prefix should be "Caption: " to match with the prompt. This was fixed when template v1.7.2 is in use.
    if isinstance(template, str):
        full_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": get_full_prompt(template, prompt, suffix).strip(),
            },
        ]
    else:
        print("**Using chat prompt**")
        assert suffix is None
        full_prompt = [*template, {"role": "user", "content": query_prefix + prompt}]
    return full_prompt


def get_full_prompt(template, prompt, suffix=None):
    assert isinstance(template, str), "Chat template requires `get_full_chat_prompt`"
    full_prompt = template.replace("{prompt}", prompt)
    if suffix:
        full_prompt = full_prompt.strip() + suffix
    return full_prompt


def get_full_model_name(model):
    if model == "gpt-3.5":
        model = "gpt-3.5-turbo"
    elif model == "vicuna":
        model = "vicuna-13b"
    elif model == "gpt-4":
        model = "gpt-4"

    return model


def get_llm_kwargs(model, template_version):
    model = get_full_model_name(model)

    print(f"Using template: {template_version}")

    template = templates[template_version]

    if (
        "vicuna" in model.lower()
        or "llama" in model.lower()
        or "freewilly" in model.lower()
    ):
        api_base = "http://localhost:8000/v1"
        max_tokens = 900
        temperature = 0.25
        headers = {}
    else:
        from utils.api_key import api_key

        api_base = "https://api.openai.com/v1"
        max_tokens = 900
        temperature = 0.25
        headers = {"Authorization": f"Bearer {api_key}"}

    llm_kwargs = EasyDict(
        model=model,
        template=template,
        api_base=api_base,
        max_tokens=max_tokens,
        temperature=temperature,
        headers=headers,
        stop=stop,
    )

    return model, llm_kwargs


def get_layout(prompt, llm_kwargs, suffix="", query_prefix="Caption: ", verbose=False):
    # No cache in this function
    model, template, api_base, max_tokens, temperature, stop, headers = (
        llm_kwargs.model,
        llm_kwargs.template,
        llm_kwargs.api_base,
        llm_kwargs.max_tokens,
        llm_kwargs.temperature,
        llm_kwargs.stop,
        llm_kwargs.headers,
    )

    if verbose:
        print("prompt:", prompt, "with suffix", suffix)

    done = False
    attempts = 0
    while not done:
        if "gpt" in model:
            r = requests.post(
                f"{api_base}/chat/completions",
                json={
                    "model": model,
                    "messages": get_full_chat_prompt(
                        template, prompt, suffix, query_prefix=query_prefix
                    ),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop if isinstance(template, str) else None,
                },
                headers=headers,
            )
        else:
            r = requests.post(
                f"{api_base}/completions",
                json={
                    "model": model,
                    "prompt": get_full_prompt(template, prompt, suffix).strip(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                },
                headers=headers,
            )

        done = r.status_code == 200

        if not done:
            print(r.json())
            attempts += 1
        if attempts >= 3 and "gpt" in model:
            print("Retrying after 1 minute")
            time.sleep(60)
        if attempts >= 5 and "gpt" in model:
            print("Exiting due to many non-successful attempts")
            exit()

    if "gpt" in model:
        if verbose > 1:
            print(f"***{r.json()}***")
        response = r.json()["choices"][0]["message"]["content"]
    else:
        response = r.json()["choices"][0]["text"]

    if verbose:
        print("resp", response)

    return response


def get_parsed_layout(*args, json_template=False, **kwargs):
    if json_template:
        return get_parsed_layout_json_resp(*args, **kwargs)
    else:
        return get_parsed_layout_text_resp(*args, **kwargs)


def get_parsed_layout_text_resp(
    prompt,
    llm_kwargs=None,
    max_partial_response_retries=1,
    override_response=None,
    strip_chars=" \t\n`",
    save_leading_text=True,
    **kwargs,
):
    """
    override_response: override the LLM response (will not query the LLM), useful for parsing existing response
    """
    if override_response is not None:
        assert (
            max_partial_response_retries == 1
        ), "override_response is specified so no partial queries are allowed"

    process_index = 0
    retries = 0
    suffix = None
    parsed_layout = {}
    reconstructed_response = ""
    while process_index < len(required_lines):
        retries += 1
        if retries > max_partial_response_retries:
            raise ValueError(
                f"Erroring due to many non-successful attempts on prompt: {prompt} with response {response}"
            )
        if override_response is not None:
            response = override_response
        else:
            response = get_layout(
                prompt, llm_kwargs=llm_kwargs, suffix=suffix, **kwargs
            )
        # print(f"Current response: {response}")
        if required_lines[process_index] in response:
            response_split = response.split(required_lines[process_index])

            # print(f"Unused leading text: {response_split[0]}")

            if save_leading_text:
                reconstructed_response += (
                    response_split[0] + required_lines[process_index]
                )
            response = response_split[1]

        while process_index < len(required_lines):
            required_line = required_lines[process_index]
            next_required_line = (
                required_lines[process_index + 1]
                if process_index + 1 < len(required_lines)
                else ""
            )
            if next_required_line in response:
                if next_required_line != "":
                    required_line_idx = response.find(next_required_line)
                    line_content = response[:required_line_idx].strip(strip_chars)
                else:
                    line_content = response.strip(strip_chars)
                if required_lines_ast[process_index]:
                    # LLMs sometimes give comments starting with " - "
                    line_content = line_content.split(" - ")[0].strip()

                    # LLMs sometimes give a list: `- name: content`
                    if line_content.startswith("-"):
                        line_content = line_content[
                            line_content.find("-") + 1 :
                        ].strip()

                    try:
                        line_content = ast.literal_eval(line_content)
                    except SyntaxError as e:
                        print(
                            f"Encountered SyntaxError with content {line_content}: {e}"
                        )
                        raise e
                parsed_layout[required_line.rstrip(":")] = line_content
                reconstructed_response += response[
                    : required_line_idx + len(next_required_line)
                ]
                response = response[required_line_idx + len(next_required_line) :]
                process_index += 1
            else:
                break
        if process_index == 0:
            # Nothing matches, retry without adding suffix
            continue
        elif process_index < len(required_lines):
            # Partial matches, add the match to the suffix and retry
            suffix = (
                "\n"
                + response.rstrip(strip_chars)
                + "\n"
                + required_lines[process_index]
            )

    parsed_layout["Prompt"] = prompt

    return parsed_layout, reconstructed_response


def get_parsed_layout_json_resp(
    prompt,
    llm_kwargs=None,
    max_partial_response_retries=1,
    override_response=None,
    strip_chars=" \t\n`",
    save_leading_text=True,
    **kwargs,
):
    """
    override_response: override the LLM response (will not query the LLM), useful for parsing existing response
    save_leading_text: ignored since we do not allow leading text in JSON
    max_partial_response_retries: ignored since we do not allow partial response in JSON
    """
    assert (
        max_partial_response_retries == 1
    ), "no partial queries are allowed in with JSON format templates"
    if override_response is not None:
        response = override_response
    else:
        response = get_layout(prompt, llm_kwargs=llm_kwargs, suffix=None, **kwargs)

    response = response.strip(strip_chars)

    # Alternatively we can use `removeprefix` in `str`.
    response = (
        response[len("Response:") :] if response.startswith("Response:") else response
    )

    response = response.strip(strip_chars)

    # print("Response:", response)

    try:
        parsed_layout = pyjson5.loads(response)
    except (
        ValueError,
        pyjson5.Json5Exception,
        pyjson5.Json5EOF,
        pyjson5.Json5DecoderException,
        pyjson5.Json5IllegalCharacter,
    ) as e:
        print(
            f"Encountered exception in parsing the response with content {response}: {e}"
        )
        raise e

    reconstructed_response = response

    parsed_layout["Prompt"] = prompt

    return parsed_layout, reconstructed_response


def get_parsed_layout_with_cache(
    prompt,
    llm_kwargs,
    verbose=False,
    max_retries=3,
    cache_miss_allowed=True,
    json_template=False,
    **kwargs,
):
    """
    Get parsed_layout with cache support. This function will only add to cache after all lines are obtained and parsed. If partial response is obtained, the model will query with the previous partial response.
    """
    # Note that cache path needs to be set correctly, as get_cache does not check whether the cache is generated with the given model in the given setting.

    response = get_cache(prompt)

    if response is not None:
        print(f"Cache hit: {prompt}")
        parsed_layout, _ = get_parsed_layout(
            prompt,
            llm_kwargs=llm_kwargs,
            max_partial_response_retries=1,
            override_response=response,
            json_template=json_template,
        )
        return parsed_layout

    print(f"Cache miss: {prompt}")

    assert cache_miss_allowed, "Cache miss is not allowed"

    done = False
    retries = 0
    while not done:
        retries += 1
        if retries >= max_retries:
            raise ValueError(
                f"Erroring due to many non-successful attempts on prompt: {prompt}"
            )
        try:
            parsed_layout, reconstructed_response = get_parsed_layout(
                prompt, llm_kwargs=llm_kwargs, json_template=json_template, **kwargs
            )
        except Exception as e:
            print(f"Error: {e}, retrying")
            traceback.print_exc()
            continue

        done = True

    add_cache(prompt, reconstructed_response)

    if verbose:
        print(f"parsed_layout = {parsed_layout}")

    return parsed_layout
