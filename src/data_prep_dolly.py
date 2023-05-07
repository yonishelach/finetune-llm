from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from datasets import Dataset, load_dataset
from functools import partial
from typing import Dict, List
from mlrun.utils import logger
import mlrun


DEFAULT_INPUT_MODEL = "EleutherAI/pythia-6.9b"
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)


def preprocess_batch(batch: Dict[str, List], tokenizer: PreTrainedTokenizer, max_length: int) -> BatchEncoding:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(path_or_dataset: str = "databricks/databricks-dolly-15k") -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset


def preprocess_dataset(tokenizer: PreTrainedTokenizer, max_length: int, seed: int = DEFAULT_SEED) -> Dataset:
    dataset = load_training_dataset()

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer


@mlrun.handler(outputs=["train_dolly_dataset:dataset", "test_dolly_dataset:dataset"])
def preprocess_dolly(
    pretrained_tokenizer_name_or_path,
    max_length: int = 1024,
    seed: int = DEFAULT_SEED,
    test_size: int = 1000,
):
    tokenizer = load_tokenizer(pretrained_tokenizer_name_or_path)
    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed)

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)
    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    return split_dataset["train"].to_pandas(), split_dataset["test"].to_pandas()
