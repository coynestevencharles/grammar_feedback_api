from langchain_openai import ChatOpenAI
import errant
from errant.annotator import Annotator as ErrantAnnotator
from errant.edit import Edit as ErrantEdit
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain.prompts import PromptTemplate
import time

from prompts import edit_extraction_prompt_v1
import config


logger = logging.getLogger(__name__)

# Global variables for loaded models/tools
edit_llm: Optional[ChatOpenAI] = None
errant_annotator: Optional[ErrantAnnotator] = None

try:
    prompt_template = PromptTemplate.from_template(edit_extraction_prompt_v1)
except Exception as e:
    logger.critical(
        f"Failed to create PromptTemplate from edit extraction prompt: {e}",
        exc_info=True,
    )
    prompt_template = None


def load_edit_llm() -> Optional[ChatOpenAI]:
    """
    Loads the Edit Refinement LLM instance once at startup.
    Returns the LLM instance or None if loading fails.
    """
    global edit_llm
    if edit_llm is None:
        try:
            logger.info("Loading Edit Refinement LLM...")
            edit_llm = ChatOpenAI(
                model=config.EDIT_LLM_NAME,
                temperature=0.0,
                max_tokens=config.EDIT_LLM_MAX_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            logger.info(
                f"Edit Refinement LLM loaded successfully: {config.EDIT_LLM_NAME}"
            )
        except Exception as e:
            logger.critical(f"Failed to load Edit Refinement LLM: {e}", exc_info=True)
            edit_llm = None
    return edit_llm


def load_errant() -> Optional[ErrantAnnotator]:
    """
    Loads the ERRANT annotator instance once at startup.
    Returns the annotator instance or None if loading fails.
    """
    global errant_annotator
    if errant_annotator is None:
        try:
            logger.info("Loading ERRANT annotator...")
            errant_annotator = errant.load("en")
            logger.info("ERRANT annotator loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load ERRANT annotator: {e}", exc_info=True)
            errant_annotator = None
    return errant_annotator


def get_rough_edits(
    sentence: str, corrected: str, response_id: str
) -> List[ErrantEdit]:
    """
    Extract ERRANT edits between original and corrected sentences.

    Args:
        sentence: The original sentence.
        corrected: The corrected sentence.
        response_id: The unique ID for logging context.

    Returns:
        A list of ERRANT Edit objects, or an empty list if ERRANT fails.
    """
    global errant_annotator
    if errant_annotator is None:
        logger.error(
            "ERRANT annotator not loaded. Cannot extract rough edits.",
            extra={"response_id": response_id},
        )
        return []

    edits: List[ErrantEdit] = []
    try:
        orig = errant_annotator.parse(sentence, tokenise=True)
        cor = errant_annotator.parse(corrected, tokenise=True)
        edits = errant_annotator.annotate(orig, cor)
        logger.debug(
            f"ERRANT found {len(edits)} rough edits.",
            extra={
                "response_id": response_id,
                "sentence": sentence,
                "corrected": corrected,
            },
        )
    except Exception as e:
        logger.error(
            f"ERRANT failed to process sentences: {e}",
            exc_info=True,
            extra={
                "response_id": response_id,
                "sentence": sentence,
                "corrected": corrected,
            },
        )
        return []

    return edits


def prepare_rough_edits_for_llm(
    edits: List[ErrantEdit], response_id: str
) -> List[Tuple[int, str, str, str]]:
    """
    Converts ERRANT edits to a simplified tuple format for the LLM prompt.

    Args:
        edits: List of ERRANT Edit objects.
        response_id: The unique ID for logging context.

    Returns:
        List of tuples: [(index, operation, original_str, corrected_str), ...].
        Returns empty list if input is empty.
    """
    rough_edit_list: List[Tuple[int, str, str, str]] = []
    if not edits:
        return rough_edit_list

    for i, edit in enumerate(edits):
        # Determine operation: M=insert, U=delete, R=replace
        operation_type = edit.type.split(":")[0] if edit.type else "UNK"

        if operation_type == "M":
            operation = "insert"
            o_str = ""
            c_str = edit.c_str
        elif operation_type == "U":
            operation = "delete"
            o_str = edit.o_str
            c_str = ""
        elif operation_type == "R":
            operation = "replace"
            o_str = edit.o_str
            c_str = edit.c_str
        else:
            logger.warning(
                f"Unknown ERRANT operation type '{edit.type}' for edit index {i}. Skipping.",
                extra={"response_id": response_id, "edit": edit},
            )
            continue

        rough_edit_list.append((i, operation, o_str, c_str))

    return rough_edit_list


async def get_refined_edits(
    sentence: str,
    corrected: str,
    errant_edits: List[ErrantEdit],
    response_id: str,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    """
    Refines "rough" ERRANT edits using an LLM.

    Args:
        sentence: The original sentence.
        corrected: The corrected sentence.
        errant_edits: The list of ERRANT Edit objects.
        response_id: The unique ID for logging context.

    Returns:
        A tuple containing:
        - List[Dict]: The list of refined edit dictionaries, including calculated spans.
        - Optional[str]: The input formatted ERRANT edits that were placed into the prompt.
        - Optional[Dict]: The raw response object from the LLM.
    """
    global edit_llm

    formatted_prompt = None
    raw_llm_response = None
    refined_edits_list = []

    # Initial Checks
    if edit_llm is None:
        logger.error(
            "Edit Refinement LLM not loaded. Cannot refine edits.",
            extra={"response_id": response_id},
        )
        return ([], None, None)
    if prompt_template is None:
        logger.error(
            "Edit extraction prompt template failed to load. Cannot format prompt.",
            extra={"response_id": response_id},
        )
        return ([], None, None)
    if not errant_edits:
        # Should not happen based on sentence-level processing logic
        logger.warning(
            "No rough ERRANT edits provided to refine.",
            extra={"response_id": response_id},
        )
        return ([], None, None)

    # Prepare Prompt
    try:
        # Prepare the simplified rough edit list for the prompt
        rough_edits_for_prompt = prepare_rough_edits_for_llm(errant_edits, response_id)
        if not rough_edits_for_prompt and errant_edits:
            logger.error(
                "Failed to prepare rough edits for LLM prompt, although ERRANT edits exist.",
                extra={"response_id": response_id},
            )
            return ([], None, None)

        # Format the list of tuples as a string for the prompt
        rough_edits_str = "[" + ", ".join(map(str, rough_edits_for_prompt)) + "]"

        # Format the final prompt
        formatted_prompt = prompt_template.format(
            source=sentence,
            corrected=corrected,
            rough_edits=rough_edits_str,
        )
        logger.debug(
            "Formatted edit refinement prompt ready.",
            extra={"response_id": response_id, "prompt": formatted_prompt},
        )

    except Exception as e:
        logger.error(
            f"Error formatting edit refinement prompt: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], None, None)

    start_time = time.time()
    try:
        response = await edit_llm.ainvoke(formatted_prompt)
        raw_llm_response = response.model_dump()
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(
            f"Edit Refinement LLM invocation completed in {processing_time:.2f} seconds.",
            extra={"response_id": response_id},
        )

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        logger.error(
            f"Error invoking Edit Refinement LLM after {processing_time:.2f} seconds: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        # Return empty list, but include the input that led to failure
        return ([], rough_edits_str, None)

    # Parse and Process Response
    try:
        result = json.loads(response.content)
        logger.debug(
            "Successfully parsed JSON response from Edit Refinement LLM.",
            extra={"response_id": response_id, "content": result},
        )

        if "refined_edits" not in result or not isinstance(
            result["refined_edits"], list
        ):
            logger.error(
                f"Invalid response structure: 'refined_edits' key missing or not a list. Content: {response.content}",
                extra={"response_id": response_id},
            )
            # Return empty list, but include the input that led to failure
            return ([], rough_edits_str, None)

        llm_refined_edits = result["refined_edits"]

        # Process each refined edit from LLM
        for i, llm_edit in enumerate(llm_refined_edits):
            # Validate the structure of each refined edit
            if not isinstance(llm_edit, dict) or not all(
                k in llm_edit
                for k in [
                    "index",
                    "action",
                    "source_words",
                    "corrected_words",
                    "attributed_edits",
                ]
            ):
                logger.warning(
                    f"Skipping malformed refined edit item at index {i}: {llm_edit}",
                    extra={"response_id": response_id},
                )
                continue

            # Get Spans based on the original errant edits attributed to this edit by the LLM
            try:
                # Pass the original errant_edits list for span calculation
                source_start, source_end, corrected_start, corrected_end = (
                    get_refined_edit_spans(errant_edits, llm_edit, response_id)
                )
                # Check if spans were successfully calculated (might return None if logic fails)
                if any(
                    span is None
                    for span in [
                        source_start,
                        source_end,
                        corrected_start,
                        corrected_end,
                    ]
                ):
                    logger.warning(
                        f"Could not determine valid spans for refined edit index {i}. Skipping.",
                        extra={"response_id": response_id, "llm_edit": llm_edit},
                    )
                    continue

            except ValueError as e:
                logger.error(
                    f"Error getting spans for refined edit index {i}: {e}",
                    exc_info=True,
                    extra={"response_id": response_id, "llm_edit": llm_edit},
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error getting spans for refined edit index {i}: {e}",
                    exc_info=True,
                    extra={"response_id": response_id, "llm_edit": llm_edit},
                )
                continue

            refined_edits_list.append(
                {
                    "index": i,
                    "action": llm_edit["action"],
                    "source_words": llm_edit["source_words"],
                    "corrected_words": llm_edit["corrected_words"],
                    "attributed_edits": llm_edit["attributed_edits"],
                    "source_start": source_start,
                    "source_end": source_end,
                    "corrected_start": corrected_start,
                    "corrected_end": corrected_end,
                }
            )

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode JSON response from Edit LLM: {e}. Content: {response.content}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], rough_edits_str, raw_llm_response)
    except Exception as e:
        logger.error(
            f"Error processing Edit LLM response content: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        return ([], rough_edits_str, raw_llm_response)

    logger.info(
        f"Edit refinement processing complete. Found {len(refined_edits_list)} refined edits.",
        extra={"response_id": response_id},
    )
    return refined_edits_list, rough_edits_str, raw_llm_response


def get_refined_edit_spans(
    errant_edits: List[ErrantEdit], llm_edit: Dict[str, Any], response_id: str
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Calculates character start/end spans in original and corrected text for a refined edit,
    based on the spans of the ERRANT edits attributed to it by the LLM.

    Args:
        errant_edits: The full list of original ERRANT Edit objects.
        llm_edit: The dictionary representing a single refined edit from the LLM.
        response_id: The unique ID for logging context.

    Returns:
        Tuple containing (source_start, source_end, corrected_start, corrected_end).
        Returns (None, None, None, None) if spans cannot be reliably determined.
    """
    source_start, source_end, corrected_start, corrected_end = None, None, None, None
    attributed_indices = llm_edit.get("attributed_edits", [])

    # Validate attributed indices
    valid_attributed_edits = []
    for idx in attributed_indices:
        if not isinstance(idx, int) or idx < 0 or idx >= len(errant_edits):
            logger.warning(
                f"Invalid attributed edit index {idx} found in refined edit. Skipping index.",
                extra={"response_id": response_id, "llm_edit": llm_edit},
            )
            continue
        valid_attributed_edits.append(errant_edits[idx])

    if not valid_attributed_edits:
        logger.warning(
            "No valid attributed ERRANT edits found for refined edit. Cannot determine spans.",
            extra={"response_id": response_id, "llm_edit": llm_edit},
        )
        return None, None, None, None

    num_attrs = len(valid_attributed_edits)
    refined_source = llm_edit.get("source_words", "")
    refined_corrected = llm_edit.get("corrected_words", "")

    try:
        if num_attrs == 1:
            rough_edit = valid_attributed_edits[0]
            o_str = rough_edit.o_str
            c_str = rough_edit.c_str

            # Case 1: Exact match (simplest)
            if o_str == refined_source and c_str == refined_corrected:
                logger.debug(
                    f"Span Case 1: Exact match.", extra={"response_id": response_id}
                )
                source_start = rough_edit.o_toks.start_char
                source_end = rough_edit.o_toks.end_char
                corrected_start = rough_edit.c_toks.start_char
                corrected_end = rough_edit.c_toks.end_char
            # Case 2: Refined edit is a subset of the rough edit
            # TODO: Consider something more robust than string.index() given the potential for multiple occurrences
            elif (refined_source and refined_source in o_str) or (
                refined_corrected and refined_corrected in c_str
            ):
                logger.debug(
                    f"Span Case 2: Subset match.", extra={"response_id": response_id}
                )
                # Source Span (if applicable)
                if refined_source and refined_source in o_str:
                    # Find start index
                    source_subset_start_offset = o_str.index(refined_source)
                    source_subset_end_offset = source_subset_start_offset + len(
                        refined_source
                    )
                    # Check for multiple occurrences (basic warning)
                    if o_str.count(refined_source) > 1:
                        logger.warning(
                            f"Refined source '{refined_source}' appears multiple times in rough source '{o_str}'. Using first occurrence for span calculation.",
                            extra={"response_id": response_id},
                        )
                    # Calculate global character offsets
                    source_start = (
                        rough_edit.o_toks.start_char + source_subset_start_offset
                    )
                    source_end = rough_edit.o_toks.start_char + source_subset_end_offset

                elif not refined_source:  # e.g., an insertion
                    source_start = (
                        rough_edit.o_toks.start_char
                    )  # Position relative to original
                    source_end = (
                        rough_edit.o_toks.start_char
                    )  # Zero-width span in original

                # Corrected Span
                if refined_corrected and refined_corrected in c_str:
                    corrected_subset_start_offset = c_str.index(refined_corrected)
                    corrected_subset_end_offset = corrected_subset_start_offset + len(
                        refined_corrected
                    )
                    if c_str.count(refined_corrected) > 1:
                        logger.warning(
                            f"Refined corrected '{refined_corrected}' appears multiple times in rough corrected '{c_str}'. Using first occurrence.",
                            extra={"response_id": response_id},
                        )
                    corrected_start = (
                        rough_edit.c_toks.start_char + corrected_subset_start_offset
                    )
                    corrected_end = (
                        rough_edit.c_toks.start_char + corrected_subset_end_offset
                    )

                elif not refined_corrected:  # e.g., a deletion
                    corrected_start = rough_edit.c_toks.start_char
                    corrected_end = (
                        rough_edit.c_toks.start_char
                    )  # Zero-width span in corrected

            else:
                # Cannot reconcile refined text with rough text
                logger.warning(
                    f"Refined edit text does not match or substring rough edit text. Cannot reliably determine subset spans.",
                    extra={
                        "response_id": response_id,
                        "refined": llm_edit,
                        "rough": rough_edit,
                    },
                )
                return None, None, None, None

        # Case 3: Multiple attributed edits
        elif num_attrs > 1:
            logger.debug(
                f"Span Case 3: Multiple ({num_attrs}) attributed edits.",
                extra={"response_id": response_id},
            )
            # Special handling for 'relocate' action
            if llm_edit.get("action") == "relocate":
                logger.debug(
                    "Handling 'relocate' action spans.",
                    extra={"response_id": response_id},
                )
                # Find the deletion part (source span) and insertion part (corrected span)
                deletion_edit = None
                insertion_edit = None

                # Prioritize explicit M/U types
                for edit in valid_attributed_edits:
                    if edit.type.startswith("U"):
                        deletion_edit = edit
                        break
                for edit in valid_attributed_edits:
                    if edit.type.startswith("M"):
                        insertion_edit = edit
                        break

                # Fallback to R types if M/U not found
                if deletion_edit is None:
                    for edit in valid_attributed_edits:
                        if (
                            edit.type.startswith("R")
                            and refined_source
                            and refined_source in edit.o_str
                        ):
                            deletion_edit = edit
                            break
                if insertion_edit is None:
                    for edit in valid_attributed_edits:
                        if (
                            edit.type.startswith("R")
                            and refined_corrected
                            and refined_corrected in edit.c_str
                        ):
                            insertion_edit = edit
                            break

                if deletion_edit and insertion_edit:
                    # Calculate source span from deletion_edit (potentially subset)
                    if deletion_edit.type.startswith("R"):
                        if refined_source and refined_source in deletion_edit.o_str:
                            source_subset_start_offset = deletion_edit.o_str.index(
                                refined_source
                            )
                            source_subset_end_offset = source_subset_start_offset + len(
                                refined_source
                            )
                            source_start = (
                                deletion_edit.o_toks.start_char
                                + source_subset_start_offset
                            )
                            source_end = (
                                deletion_edit.o_toks.start_char
                                + source_subset_end_offset
                            )
                        else:
                            logger.warning(
                                "Relocate source text not found in fallback R edit.",
                                extra={"response_id": response_id},
                            )
                    else:  # U type
                        source_start = deletion_edit.o_toks.start_char
                        source_end = deletion_edit.o_toks.end_char

                    # Calculate corrected span from insertion_edit (potentially subset)
                    if insertion_edit.type.startswith("R"):
                        if (
                            refined_corrected
                            and refined_corrected in insertion_edit.c_str
                        ):
                            corrected_subset_start_offset = insertion_edit.c_str.index(
                                refined_corrected
                            )
                            corrected_subset_end_offset = (
                                corrected_subset_start_offset + len(refined_corrected)
                            )
                            corrected_start = (
                                insertion_edit.c_toks.start_char
                                + corrected_subset_start_offset
                            )
                            corrected_end = (
                                insertion_edit.c_toks.start_char
                                + corrected_subset_end_offset
                            )
                        else:
                            logger.warning(
                                "Relocate corrected text not found in fallback R edit.",
                                extra={"response_id": response_id},
                            )
                    else:  # M type
                        corrected_start = insertion_edit.c_toks.start_char
                        corrected_end = insertion_edit.c_toks.end_char
                else:
                    logger.warning(
                        "Could not identify distinct deletion and insertion components for relocate action.",
                        extra={"response_id": response_id, "llm_edit": llm_edit},
                    )
                    return None, None, None, None

            else:  # General multi-attribution (not relocate) - Use heuristic: min/max spans
                logger.debug(
                    "Handling general multi-attribution spans using min/max heuristic.",
                    extra={"response_id": response_id},
                )
                # This might cover a wider span than the actual refined change.
                # Source span (only consider edits that have original text)
                o_spans = [
                    (e.o_toks.start_char, e.o_toks.end_char)
                    for e in valid_attributed_edits
                    if e.o_toks
                ]
                if o_spans:
                    source_start = min(s[0] for s in o_spans)
                    source_end = max(s[1] for s in o_spans)

                # Corrected span (only consider edits that have corrected text)
                c_spans = [
                    (e.c_toks.start_char, e.c_toks.end_char)
                    for e in valid_attributed_edits
                    if e.c_toks
                ]
                if c_spans:
                    corrected_start = min(s[0] for s in c_spans)
                    corrected_end = max(s[1] for s in c_spans)

        # Check if all necessary spans were assigned
        if any(
            s is None
            for s in [source_start, source_end, corrected_start, corrected_end]
        ):
            logger.warning(
                f"Failed to determine one or more spans.",
                extra={
                    "response_id": response_id,
                    "spans": [source_start, source_end, corrected_start, corrected_end],
                },
            )
            return None, None, None, None

        return (
            int(source_start),
            int(source_end),
            int(corrected_start),
            int(corrected_end),
        )

    except Exception as e:
        logger.error(
            f"Unexpected error in get_refined_edit_spans: {e}",
            exc_info=True,
            extra={"response_id": response_id, "llm_edit": llm_edit},
        )
        return None, None, None, None
