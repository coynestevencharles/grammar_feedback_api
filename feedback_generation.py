from langchain_openai import ChatOpenAI
import json
from typing import List, Tuple, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
import logging
import time
import os
from errant.edit import Edit as ErrantEdit

from api_models import feedbackComment
from prompts import feedback_prompt_v1
import config

logger = logging.getLogger(__name__)

feedback_llm: Optional[ChatOpenAI] = None
errant_feedback_templates: Dict[str, Any] = {}

try:
    errant_feedback_templates_file = os.path.join(
        config.DATA_DIR, "errant_feedback_templates.json"
    )
    logger.info(
        f"Loading ERRANT feedback templates from: {errant_feedback_templates_file}"
    )
    with open(errant_feedback_templates_file, "r") as f:
        errant_feedback_templates = json.load(f)
    logger.info("ERRANT feedback templates loaded successfully.")
except FileNotFoundError:
    logger.critical(
        f"ERRANT feedback templates file not found at {errant_feedback_templates_file}."
    )
except json.JSONDecodeError as e:
    logger.critical(
        f"Error decoding JSON from ERRANT feedback templates file {errant_feedback_templates_file}: {e}",
        exc_info=True,
    )
except Exception as e:
    logger.critical(
        f"An unexpected error occurred loading ERRANT feedback templates: {e}",
        exc_info=True,
    )


# Load LLM
def load_feedback_llm() -> Optional[ChatOpenAI]:
    """
    Loads the Feedback LLM instance once at startup.
    Returns the LLM instance or None if loading fails.
    """
    global feedback_llm
    if feedback_llm is None:
        try:
            feedback_llm = ChatOpenAI(
                model=config.FEEDBACK_LLM_NAME,
                temperature=0.0,
                max_tokens=config.FEEDBACK_LLM_MAX_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info(f"Feedback LLM loaded successfully: {config.FEEDBACK_LLM_NAME}")
        except Exception as e:
            logger.critical(f"Failed to load Feedback LLM: {e}", exc_info=True)
            feedback_llm = None
    return feedback_llm


def emphasize_sentences(
    source: str,
    corrected: str,
    source_start: int,
    source_end: int,
    corrected_start: int,
    corrected_end: int,
) -> Tuple[str, str]:
    """Emphasize the target edit in the source and corrected sentences using markdown."""
    # Source emphasis
    if source_start == source_end:  # Zero-width span (insertion point)
        marker_pos = min(source_end, len(source))
        highlighted_source = (
            source[:marker_pos] + "**" + source[marker_pos:]
        )  # Use space for visibility
    elif source_start < source_end:  # Normal span
        highlighted_source = (
            source[:source_start]
            + "*"
            + source[source_start:source_end]
            + "*"
            + source[source_end:]
        )
    else:  # Invalid span
        highlighted_source = source

    # Corrected emphasis
    if corrected_start == corrected_end:  # Zero-width span (deletion point)
        marker_pos = min(corrected_end, len(corrected))
        highlighted_corrected = corrected[:marker_pos] + "**" + corrected[marker_pos:]
    elif corrected_start < corrected_end:  # Normal span
        highlighted_corrected = (
            corrected[:corrected_start]
            + "*"
            + corrected[corrected_start:corrected_end]
            + "*"
            + corrected[corrected_end:]
        )
    else:  # Invalid span
        highlighted_corrected = corrected

    return highlighted_source, highlighted_corrected


def edit_to_string(target_edit: Dict[str, Any], response_id: str) -> Optional[str]:
    """Convert the target refined edit dictionary to a string summary."""
    action = target_edit.get("action")
    source_words = target_edit.get("source_words", "")
    corrected_words = target_edit.get("corrected_words", "")

    if action == "replace":
        target_edit_str = (
            f"The replacement of '{source_words}' with '{corrected_words}'"
        )
    elif action == "insert":
        target_edit_str = f"The insertion of '{corrected_words}'"
    elif action == "delete":
        target_edit_str = f"The deletion of '{source_words}'"
    elif action == "relocate":
        if source_words != corrected_words:
            target_edit_str = (
                f"The relocation of '{source_words}' (becoming '{corrected_words}')"
            )
        else:
            target_edit_str = f"The relocation of '{source_words}'"
    else:
        logger.warning(
            f"Unknown action '{action}' in target_edit. Cannot create string.",
            extra={"response_id": response_id, "edit": target_edit},
        )
        return None
    return target_edit_str


def construct_feedback_input_batch(
    emphasized_source_sentences: List[str],
    emphasized_corrected_sentences: List[str],
    target_edit_strs: List[str],
    response_id: str,
) -> str:
    """Construct the multi-item input string for the feedback prompt."""
    input_batch_items = []
    # Ensure all lists have the same length
    if not (
        len(emphasized_source_sentences)
        == len(emphasized_corrected_sentences)
        == len(target_edit_strs)
    ):
        logger.error(
            "Mismatch in lengths of lists for constructing feedback input batch.",
            extra={"response_id": response_id},
        )
        return ""

    for i in range(len(emphasized_source_sentences)):
        if target_edit_strs[i] is None:
            logger.warning(
                f"Skipping item {i} in feedback input batch due to None target_edit_str.",
                extra={"response_id": response_id},
            )
            continue

        input_batch_items.append(
            f"index: {i}\n"
            f"original_sentence: {emphasized_source_sentences[i]}\n"
            f"corrected_sentence: {emphasized_corrected_sentences[i]}\n"
            f"target_edit: {target_edit_strs[i]}"
        )
    return "\n\n".join(input_batch_items)


async def get_llm_feedback(
    sentence: str, corrected: str, refined_edits: List[Dict[str, Any]], response_id: str
) -> Tuple[List[feedbackComment], Optional[str], Optional[Dict[str, Any]]]:
    """
    Generates feedback comments for refined edits using an LLM.

    Args:
        sentence: The original sentence.
        corrected: The corrected sentence.
        refined_edits: List of refined edit dictionaries (output from get_refined_edits).
        response_id: The unique ID for logging context.

    Returns:
        A tuple containing:
        - List[feedbackComment]: List of Pydantic feedback comment objects.
        - Optional[str]: The input batch string sent to the LLM.
        - Optional[Dict]: The raw response object from the LLM.
    """
    global feedback_llm

    feedback_list = []
    input_batch_str = None
    raw_llm_response = None
    formatted_prompt = None

    # Initial Checks
    if feedback_llm is None:
        logger.error(
            "Feedback LLM not loaded. Cannot generate LLM feedback.",
            extra={"response_id": response_id},
        )
        return ([], None, None)
    if not refined_edits:
        logger.info(
            "No refined edits provided. No LLM feedback to generate.",
            extra={"response_id": response_id},
        )
        return ([], None, None)
    try:
        current_prompt_template = PromptTemplate.from_template(feedback_prompt_v1)
    except Exception as e:
        logger.error(
            f"Feedback prompt template failed to load or is invalid: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], None, None)

    # Prepare Inputs for Batch
    target_edit_strs: List[Optional[str]] = []
    emphasized_source_sentences: List[str] = []
    emphasized_corrected_sentences: List[str] = []
    valid_edits_indices: List[int] = []

    for i, edit in enumerate(refined_edits):
        try:
            source_start = int(edit["source_start"])
            source_end = int(edit["source_end"])
            corrected_start = int(edit["corrected_start"])
            corrected_end = int(edit["corrected_end"])
        except (TypeError, ValueError, KeyError) as e:
            logger.warning(
                f"Invalid or missing span in refined edit at index {i}. Skipping edit for feedback generation.",
                extra={"response_id": response_id, "edit": edit, "error": str(e)},
            )
            continue

        edit_str = edit_to_string(edit, response_id)
        if edit_str is None:
            continue

        emphasized_source, emphasized_corrected = emphasize_sentences(
            sentence,
            corrected,
            source_start,
            source_end,
            corrected_start,
            corrected_end,
        )

        target_edit_strs.append(edit_str)
        emphasized_source_sentences.append(emphasized_source)
        emphasized_corrected_sentences.append(emphasized_corrected)
        valid_edits_indices.append(i)

    if not target_edit_strs:
        logger.warning(
            "No valid refined edits remaining after preparing inputs for feedback.",
            extra={"response_id": response_id},
        )
        return ([], None, None)

    # Construct Input Batch and Format Prompt
    try:
        input_batch_str = construct_feedback_input_batch(
            emphasized_source_sentences,
            emphasized_corrected_sentences,
            target_edit_strs,
            response_id,
        )
        if not input_batch_str:
            logger.error(
                "Constructed input batch string is empty.",
                extra={"response_id": response_id},
            )
            return ([], None, None)

        formatted_prompt = current_prompt_template.format(input_batch=input_batch_str)
        logger.debug(
            "Formatted feedback prompt ready.", extra={"response_id": response_id}
        )

    except Exception as e:
        logger.error(
            f"Error formatting feedback prompt: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return (
            [],
            input_batch_str,
            None,
        )

    # Invoke LLM
    start_time = time.time()
    try:
        response: AIMessage = await feedback_llm.ainvoke(formatted_prompt)
        raw_llm_response = response.model_dump()
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(
            f"Feedback LLM invocation completed in {processing_time:.2f} seconds.",
            extra={"response_id": response_id},
        )
        if (
            raw_llm_response
            and "usage_metadata" in raw_llm_response
            and raw_llm_response["usage_metadata"]
        ):
            logger.info(
                "Feedback LLM token usage.",
                extra={
                    "response_id": response_id,
                    "usage": raw_llm_response["usage_metadata"],
                },
            )

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(
            f"Error invoking Feedback LLM after {processing_time:.2f} seconds: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], input_batch_str, None)

    # Parse and Validate Response
    try:
        result = json.loads(response.content)
        logger.debug(
            "Successfully parsed JSON response from Feedback LLM.",
            extra={"response_id": response_id, "content": result},
        )

        if "feedback_list" not in result or not isinstance(
            result["feedback_list"], list
        ):
            logger.error(
                f"Invalid response structure: 'feedback_list' key missing or not a list. Content: {response.content}",
                extra={"response_id": response_id},
            )
            return ([], input_batch_str, raw_llm_response)

        llm_feedback_items = result["feedback_list"]

        # validate but move forward with partially valid results
        num_expected_feedback = len(target_edit_strs)
        if len(llm_feedback_items) != num_expected_feedback:
            logger.warning(
                f"Number of feedback items from LLM ({len(llm_feedback_items)}) does not match number of valid inputs sent ({num_expected_feedback}). Proceeding with available items.",
                extra={"response_id": response_id},
            )

        processed_indices = set()
        for llm_feedback in llm_feedback_items:
            if not isinstance(llm_feedback, dict):
                logger.warning(
                    f"Skipping non-dict item in feedback_list: {llm_feedback}",
                    extra={"response_id": response_id},
                )
                continue

            # Validate required keys
            required_keys = [
                "index",
                "highlighted_sentence",
                "error_tag",
                "feedback_explanation",
                "feedback_suggestion",
            ]
            if not all(key in llm_feedback for key in required_keys):
                logger.warning(
                    f"Skipping feedback item missing required keys: {llm_feedback}",
                    extra={"response_id": response_id},
                )
                continue

            item_index = llm_feedback.get("index")
            if not isinstance(item_index, int) or not (
                0 <= item_index < num_expected_feedback
            ):
                logger.warning(
                    f"Skipping feedback item with invalid index ({item_index}). Expected 0-{num_expected_feedback-1}.",
                    extra={"response_id": response_id, "item": llm_feedback},
                )
                continue
            if item_index in processed_indices:
                logger.warning(
                    f"Skipping feedback item with duplicate index ({item_index}).",
                    extra={"response_id": response_id, "item": llm_feedback},
                )
                continue
            processed_indices.add(item_index)

            # Calculate Spans from LLM's Highlight Markers
            highlighted_sentence = llm_feedback["highlighted_sentence"]
            # Clean asterisks
            no_asterisks = highlighted_sentence.replace("*", "").replace("**", "")
            # Attempt to find unique markers < >
            highlight_start = no_asterisks.find("<")
            # Find the closing marker *after* the opening one
            highlight_end = (
                no_asterisks.find(">", highlight_start + 1)
                if highlight_start != -1
                else -1
            )

            if highlight_start == -1 or highlight_end == -1:
                logger.warning(
                    f"Could not find valid highlight markers '< >' in LLM output sentence for index {item_index}. Cannot determine highlight span.",
                    extra={"response_id": response_id, "hl_sent": highlighted_sentence},
                )
                continue

            # Remove markers to get clean text for comparison and span calculation
            cleaned_for_comparison = no_asterisks.replace("<", "").replace(">", "")

            # Compare cleaned text to original sentence
            if cleaned_for_comparison != sentence:
                logger.warning(
                    f"Cleaned highlighted sentence from LLM does not match original sentence for index {item_index}. Spans may be unreliable. Original: '{sentence}', Highlighted: '{highlighted_sentence}', Cleaned: '{cleaned_for_comparison}'",
                    extra={"response_id": response_id},
                )
                # Skip this feedback
                continue

            # Adjust end index of > for accurate span
            highlight_end_adjusted = highlight_end - 1

            highlight_text = sentence[highlight_start:highlight_end_adjusted]

            feedback_data = dict(
                source=sentence,
                corrected=corrected,
                highlight_start=highlight_start,
                highlight_end=highlight_end_adjusted,
                highlight_text=highlight_text,
                error_tag=llm_feedback["error_tag"],
                feedback_explanation=llm_feedback["feedback_explanation"],
                feedback_suggestion=llm_feedback["feedback_suggestion"],
            )

            feedback_list.append(feedback_data)

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response from Feedback LLM: {e}. Content: {response.content}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], input_batch_str, raw_llm_response)
    except Exception as e:
        logger.error(
            f"Error processing Feedback LLM response content: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], input_batch_str, raw_llm_response)

    logger.info(
        f"LLM feedback generation complete. Generated {len(feedback_list)} feedback items.",
        extra={"response_id": response_id},
    )
    return feedback_list, input_batch_str, raw_llm_response


# Rule-Based Feedback Generation


async def get_rule_based_feedback(
    sentence: str, corrected: str, rough_edit: ErrantEdit, response_id: str
) -> Optional[feedbackComment]:
    """
    Generates feedback for a single rough ERRANT edit using predefined templates.

    Args:
        sentence: The original sentence.
        corrected: The corrected sentence.
        rough_edit: The single ERRANT Edit object.
        response_id: The unique ID for logging context.

    Returns:
        A feedbackComment object or None if feedback cannot be generated.
    """
    if not errant_feedback_templates:
        logger.error(
            "ERRANT feedback templates not loaded. Cannot generate rule-based feedback.",
            extra={"response_id": response_id},
        )
        return None

    try:
        source_start = rough_edit.o_toks.start_char
        source_end = rough_edit.o_toks.end_char
        source_str = rough_edit.o_str
        # NOTE: Not used now, but might be in a future version where we show corrected text
        corrected_start = rough_edit.c_toks.start_char
        corrected_end = rough_edit.c_toks.end_char
        corrected_str = rough_edit.c_str

        error_type = rough_edit.type
        if not error_type or ":" not in error_type:
            logger.warning(
                f"Invalid or missing ERRANT error type '{error_type}'.",
                extra={"response_id": response_id, "edit": rough_edit},
            )
            return None
        op_type, pos_type = error_type.split(":", 1)

        feedback_explanation = None
        feedback_suggestion = None
        title = None

        # Template Lookup Logic
        selected_feedback_templates = errant_feedback_templates.get(pos_type, {})
        title = selected_feedback_templates.get("title")

        if pos_type == "ORTH":
            # Check casing issues
            if source_str.lower() == corrected_str.lower():
                if (
                    corrected_str
                    and source_str
                    and corrected_str[0].isupper()
                    and source_str[0].islower()
                ):
                    feedback_explanation = (
                        f'It seems "{source_str}" should be capitalized.'
                    )
                    feedback_suggestion = (
                        f'Consider capitalizing it: "{corrected_str}".'
                    )
                elif (
                    corrected_str
                    and source_str
                    and corrected_str[0].islower()
                    and source_str[0].isupper()
                ):
                    feedback_explanation = (
                        f'The word "{source_str}" seems incorrectly capitalized.'
                    )
                    feedback_suggestion = f'Consider lowercase: "{corrected_str}".'
                else:
                    feedback_explanation = (
                        f'Check the capitalization of "{source_str}".'
                    )
                    feedback_suggestion = f'Perhaps use "{corrected_str}"?'
            # Check spacing issues
            elif (
                len(source_str.replace(" ", "")) == len(corrected_str)
                and " " in source_str
            ):  # Join words
                feedback_explanation = (
                    f'"{source_str}" might need to be joined together.'
                )
                feedback_suggestion = f'Consider writing it as "{corrected_str}".'
            elif (
                len(source_str) == len(corrected_str.replace(" ", ""))
                and " " in corrected_str
            ):  # Split words
                feedback_explanation = f'It seems "{source_str}" should be split.'
                feedback_suggestion = f'Try writing it as "{corrected_str}".'

        # General template lookup for other POS types
        else:
            template_obj = selected_feedback_templates.get(op_type)
            # Fallback to Default template if specific not found
            if not isinstance(template_obj, dict):
                default_templates = errant_feedback_templates.get("Default", {})
                template_obj = default_templates.get(op_type)

            if isinstance(template_obj, dict):
                feedback_explanation = template_obj.get("feedback_explanation")
                feedback_suggestion = template_obj.get("feedback_suggestion")
            else:
                logger.warning(
                    f"No valid feedback template found for error type {error_type} (POS: {pos_type}, Op: {op_type}).",
                    extra={"response_id": response_id},
                )
                return None

        # Clean strings for PUNCT template filling
        display_source_str = source_str
        display_corrected_str = corrected_str
        if pos_type == "PUNCT":
            display_source_str = "".join(
                [c for c in source_str if not c.isalnum() and not c.isspace()]
            )
            display_corrected_str = "".join(
                [c for c in corrected_str if not c.isalnum() and not c.isspace()]
            )

        # Fill Templates
        final_explanation = "Error generating explanation."
        final_suggestion = "Error generating suggestion."
        if feedback_explanation:
            try:
                final_explanation = feedback_explanation.replace(
                    "{source_str}", display_source_str
                ).replace("{corrected_str}", display_corrected_str)
            except Exception as fmt_e:
                logger.error(
                    f"Error formatting explanation template for {error_type}: {fmt_e}",
                    exc_info=True,
                    extra={"response_id": response_id},
                )
        if feedback_suggestion:
            try:
                final_suggestion = feedback_suggestion.replace(
                    "{source_str}", display_source_str
                ).replace("{corrected_str}", display_corrected_str)
            except Exception as fmt_e:
                logger.error(
                    f"Error formatting suggestion template for {error_type}: {fmt_e}",
                    exc_info=True,
                    extra={"response_id": response_id},
                )

        # Adjust Highlight Span for Insertions
        highlight_start = source_start
        highlight_end = source_end
        if op_type == "M" and source_start == source_end:  # Insertion
            try:
                highlight_start, highlight_end = extend_insertion_highlight(
                    sentence, source_start
                )
                logger.debug(
                    f"Extended insertion highlight for {error_type} from {source_start} to {highlight_start}-{highlight_end}",
                    extra={"response_id": response_id},
                )
            except Exception as ext_e:
                logger.error(
                    f"Error extending insertion highlight: {ext_e}",
                    exc_info=True,
                    extra={"response_id": response_id},
                )
                # Keep original zero-width span if extension fails

        # Determine Error Tag
        # Use title from template if available, otherwise fallback to error type
        error_tag = title if title else error_type
        if not title:
            logger.warning(
                f"No title found for error type '{error_type}', using type as tag.",
                extra={"response_id": response_id},
            )

        # Create Feedback Object
        highlight_text = sentence[highlight_start:highlight_end]

        feedback_data = dict(
            source=sentence,
            corrected=corrected,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
            highlight_text=highlight_text,
            error_tag=error_tag,
            feedback_explanation=final_explanation,
            feedback_suggestion=final_suggestion,
        )

        return feedback_data

    except Exception as e:
        # Catch-all for unexpected errors during rule-based generation
        logger.error(
            f"Unexpected error generating rule-based feedback for edit {rough_edit}: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return None


# Helper for Insertion Highlighting
def extend_insertion_highlight(text: str, insertion_point: int) -> tuple[int, int]:
    """Extends a zero-width span (insertion point) to include adjacent words/tokens."""
    n = len(text)
    if not (0 <= insertion_point <= n):
        logger.error(
            f"Insertion point {insertion_point} out of bounds for text length {n}."
        )
        # Fallback to zero-width span
        return insertion_point, insertion_point

    new_span_start = insertion_point
    new_span_end = insertion_point

    # Find left boundary
    current_pos = insertion_point - 1
    while current_pos >= 0 and text[current_pos].isspace():
        current_pos -= 1
    start_of_left_token = current_pos
    while start_of_left_token >= 0 and not text[start_of_left_token].isspace():
        start_of_left_token -= 1
    if current_pos > start_of_left_token:
        new_span_start = start_of_left_token + 1

    # Find right boundary
    current_pos = insertion_point
    while current_pos < n and text[current_pos].isspace():
        current_pos += 1
    end_of_right_token = current_pos
    while end_of_right_token < n and not text[end_of_right_token].isspace():
        end_of_right_token += 1
    if end_of_right_token > current_pos:
        new_span_end = end_of_right_token

    final_start = max(0, new_span_start)
    final_end = min(n, new_span_end)

    # Ensure start <= end
    if final_start > final_end:
        logger.warning(
            f"Highlight extension resulted in start > end ({final_start} > {final_end}). Using zero-width span at {insertion_point}."
        )
        return insertion_point, insertion_point

    return final_start, final_end
