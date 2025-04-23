from typing import List, Tuple, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
import json
import time
import os

from prompts import correction_prompt_v1
import config

logger = logging.getLogger(__name__)

correction_llm: Optional[ChatOpenAI] = None

try:
    prompt_template = PromptTemplate.from_template(correction_prompt_v1)
except Exception as e:
    logger.critical(
        f"Failed to create PromptTemplate from correction_prompt_v1: {e}", exc_info=True
    )
    prompt_template = None


def load_correction_llm() -> Optional[ChatOpenAI]:
    """
    Loads the Correction LLM instance once at startup.
    Returns the LLM instance or None if loading fails.
    """
    global correction_llm
    if correction_llm is None:
        try:
            correction_llm = ChatOpenAI(
                model=config.CORRECTION_LLM_NAME,
                temperature=0.0,
                max_tokens=config.CORRECTION_LLM_MAX_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info(
                f"Correction LLM loaded successfully: {config.CORRECTION_LLM_NAME}"
            )
        except Exception as e:
            logger.critical(f"Failed to load Correction LLM: {e}", exc_info=True)
            correction_llm = None
    return correction_llm


def format_sentences_for_prompt(sentences: List[str]) -> str:
    """Formats sentences with indices for the LLM prompt."""
    indexed_lines = [f"{i}. {sentence}" for i, sentence in enumerate(sentences)]
    return "\n".join(indexed_lines)


async def get_corrections(
    sentences: List[str], response_id: str
) -> Tuple[List[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Gets corrections for a list of sentences using the loaded LLM.

    Args:
        sentences: The list of original sentences.
        response_id: The unique ID for this request for logging.

    Returns:
        A tuple containing:
        - List[str]: The list of corrected sentences (or original if correction failed/not needed).
            Length should match the input list.
        - Optional[str]: The input sentences formatted for the prompt.
        - Optional[Dict[str, Any]]: The raw response object from the LLM.
            None if the LLM call failed before getting a response object.
    """
    global correction_llm

    if correction_llm is None:
        logger.error(
            "Correction LLM not loaded. Cannot perform correction.",
            extra={"response_id": response_id},
        )
        # Return structure indicating failure, matching the type hint
        return (
            [s for s in sentences],
            None,
            None,
        )

    if prompt_template is None:
        logger.error(
            "Correction prompt template failed to load. Cannot format prompt.",
            extra={"response_id": response_id},
        )
        return ([s for s in sentences], None, None)

    if not sentences:
        logger.warning(
            "No sentences provided for correction.", extra={"response_id": response_id}
        )
        return ([], None, None)

    if not isinstance(sentences, list):
        logger.error(
            f"Input 'sentences' is not a list (type: {type(sentences)}). Cannot process.",
            extra={"response_id": response_id},
        )
        return ([], None, None)

    indexed_sentences_string = None
    formatted_prompt = None
    raw_llm_response = None
    corrected_sentences_list = [None] * len(sentences)

    try:
        indexed_sentences_string = format_sentences_for_prompt(sentences)
        if not indexed_sentences_string and sentences:
            logger.warning(
                "Formatting sentences resulted in empty string despite non-empty input.",
                extra={"response_id": response_id},
            )
            return ([s for s in sentences], None, None)

        formatted_prompt = prompt_template.format(
            formatted_sentences=indexed_sentences_string
        )
        logger.debug(
            f"Formatted GEC prompt ready.",
            extra={"response_id": response_id, "prompt": formatted_prompt},
        )

    except Exception as e:
        logger.error(
            f"Error formatting correction prompt: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([s for s in sentences], None, None)

    # Invoke LLM
    start_time = time.time()
    try:
        response: AIMessage = await correction_llm.ainvoke(formatted_prompt)
        raw_llm_response = response.model_dump()

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(
            f"Correction LLM invocation completed in {processing_time:.2f} seconds.",
            extra={"response_id": response_id},
        )

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(
            f"Error invoking Correction LLM after {processing_time:.2f} seconds: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([s for s in sentences], indexed_sentences_string, None)

    # Parse and Validate Response
    try:
        content_dict = json.loads(response.content)
        logger.debug(
            "Successfully parsed JSON response from Correction LLM.",
            extra={"response_id": response_id, "content": content_dict},
        )

        # Validate structure
        if "corrected" not in content_dict or not isinstance(
            content_dict["corrected"], list
        ):
            logger.error(
                f"Invalid response structure: 'corrected' key missing or not a list. Content: {response.content}",
                extra={"response_id": response_id},
            )
            # Fallback to original sentences, return indexed_sentences_string and raw response for analysis
            return ([s for s in sentences], indexed_sentences_string, raw_llm_response)

        corrected_data = content_dict["corrected"]

        # Check if the number of items matches the input sentences
        if len(corrected_data) != len(sentences):
            logger.warning(
                f"Number of corrected items ({len(corrected_data)}) does not match input sentences ({len(sentences)}).",
                extra={"response_id": response_id},
            )

            # Revert to original sentences if mismatch
            # TODO: Determine if there's anything better we can do
            corrected_sentences_list = [s for s in sentences]

        processed_indices = set()
        for item in corrected_data:
            if not isinstance(item, list) or len(item) != 2:
                logger.warning(
                    f"Skipping invalid item format in 'corrected' list: {item}",
                    extra={"response_id": response_id},
                )
                continue

            idx, corrected_text = item[0], item[1]

            # Index validation
            if not isinstance(idx, int):
                logger.warning(
                    f"Skipping item with non-integer index: {item}",
                    extra={"response_id": response_id},
                )
                continue
            if idx < 0 or idx >= len(sentences):
                logger.warning(
                    f"Skipping item with out-of-range index ({idx}): {item}",
                    extra={"response_id": response_id},
                )
                continue
            if idx in processed_indices:
                logger.warning(
                    f"Skipping item with duplicate index ({idx}): {item}",
                    extra={"response_id": response_id},
                )
                continue

            # Correction Text Validation
            original_sentence = sentences[idx]
            if corrected_text is None:
                # LLM indicates no correction needed
                corrected_sentences_list[idx] = original_sentence
                logger.debug(
                    f"Using original sentence for index {idx} as correction was None.",
                    extra={"response_id": response_id},
                )
            elif isinstance(corrected_text, str):
                # Valid correction string
                if (
                    original_sentence and not corrected_text
                ):  # Check for empty correction when original wasn't empty
                    logger.warning(
                        f"Corrected sentence for index {idx} is empty, using original. Source: '{original_sentence}'",
                        extra={"response_id": response_id},
                    )
                    corrected_sentences_list[idx] = original_sentence
                else:
                    corrected_sentences_list[idx] = corrected_text
            else:
                # Invalid type for corrected text
                logger.warning(
                    f"Corrected sentence for index {idx} is not a string or None (type: {type(corrected_text)}), using original.",
                    extra={"response_id": response_id},
                )
                corrected_sentences_list[idx] = original_sentence

            processed_indices.add(idx)

        # Fill in any missing indices with original sentences
        missing_count = 0
        for i in range(len(sentences)):
            if i not in processed_indices:
                corrected_sentences_list[i] = sentences[i]
                missing_count += 1
        if missing_count > 0:
            logger.warning(
                f"{missing_count} indices were missing from the 'corrected' response list; originals used.",
                extra={"response_id": response_id},
            )

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response from LLM: {e}. Content: {response.content}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([s for s in sentences], indexed_sentences_string, raw_llm_response)
    except Exception as e:
        # Catch other potential errors during validation
        logger.error(
            f"Error processing LLM response content: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([s for s in sentences], indexed_sentences_string, raw_llm_response)

    logger.debug(
        f"Correction processing complete. Input sentences: {len(sentences)}, Output sentences: {len(corrected_sentences_list)}.",
        extra={"response_id": response_id},
    )
    # Final length check just in case all the above fails
    if len(corrected_sentences_list) != len(sentences):
        logger.error(
            f"FINAL LENGTH MISMATCH: Input {len(sentences)}, Output {len(corrected_sentences_list)}.",
            extra={"response_id": response_id},
        )
        return (
            [s for s in sentences],
            indexed_sentences_string,
            raw_llm_response,
        )

    return corrected_sentences_list, indexed_sentences_string, raw_llm_response
