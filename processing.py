import time
import traceback
import asyncio
from datetime import datetime, UTC
import logging
import config
from edit_extraction import get_rough_edits, get_refined_edits
from feedback_generation import get_rule_based_feedback, get_llm_feedback

logger = logging.getLogger(__name__)


async def process_sentence_pair(
    pair: dict, response_id: str, feedback_system: str, sentence_index: int
) -> tuple[dict, dict]:
    """
    Processes one sentence pair using selected system.

    Args:
        - pair: Dictionary containing the original and corrected sentences.
        - response_id: Unique identifier for the response.
        - feedback_system: The system used for feedback generation (LLM or rule-based).
        - sentence_index: Index of the sentence in the batch.

    Returns:
        - processing_result: Dictionary containing the original sentence, corrected sentence, and feedback list.
        - sentence_analysis: Dictionary containing detailed analysis of the processing steps and errors.
    """
    start_time_sentence = time.time()
    sentence_analysis = {
        "sentence_index": sentence_index,
        "original_sentence": pair["sentence"],
        "corrected_sentence": pair["corrected"],
        "feedback_system": feedback_system,
        "steps": [],
        "errors": [],
        "result_feedback_count": 0,
    }

    processing_result = {
        "sentence": pair["sentence"],
        "corrected": pair["corrected"],
        "feedback_list": [],
    }

    rough_edits, refined_edits = [], []
    edit_formatted_input, edit_response = None, None
    feedback_formatted_input, feedback_response = None, None

    try:
        step_start_time = time.time()
        try:
            rough_edits = get_rough_edits(
                sentence=pair["sentence"],
                corrected=pair["corrected"],
                response_id=response_id,
            )
            sentence_analysis["steps"].append(
                {
                    "step": "get_rough_edits",
                    "timestamp": datetime.now(UTC),
                    "duration_sec": time.time() - step_start_time,
                    "status": "success",
                    "output_edit_count": len(rough_edits),
                    "output_edits": rough_edits,
                }
            )
        except Exception as e:
            logger.error(
                f"Error extracting rough edits",
                exc_info=True,
                extra={"response_id": response_id, "sentence_index": sentence_index},
            )
            sentence_analysis["errors"].append(
                {
                    "step": "get_rough_edits",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            sentence_analysis["steps"].append(
                {
                    "step": "get_rough_edits",
                    "status": "error",
                    "duration_sec": time.time() - step_start_time,
                }
            )
            rough_edits = []

        if rough_edits:
            # Refined Edit Extraction
            if feedback_system == config.LLM_BASED_SYSTEM:
                step_start_time = time.time()
                try:
                    refined_edits, edit_formatted_input, edit_response = (
                        await get_refined_edits(
                            pair["sentence"], pair["corrected"], rough_edits, response_id
                        )
                    )
                    sentence_analysis["steps"].append(
                        {
                            "step": "get_refined_edits",
                            "timestamp": datetime.now(UTC),
                            "duration_sec": time.time() - step_start_time,
                            "status": "success",
                            "llm_input": edit_formatted_input,
                            "llm_output": edit_response,
                            "output_edit_count": len(refined_edits),
                            "output_edits": refined_edits,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error refining edits",
                        exc_info=True,
                        extra={
                            "response_id": response_id,
                            "sentence_index": sentence_index,
                        },
                    )
                    sentence_analysis["errors"].append(
                        {
                            "step": "get_refined_edits",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    sentence_analysis["steps"].append(
                        {
                            "step": "get_refined_edits",
                            "status": "error",
                            "duration_sec": time.time() - step_start_time,
                        }
                    )
                    refined_edits = []

                # LLM Feedback Generation
                if refined_edits:
                    step_start_time = time.time()
                    try:
                        feedback_list, feedback_formatted_input, feedback_response = (
                            await get_llm_feedback(
                                pair["sentence"], pair["corrected"], refined_edits, response_id
                            )
                        )
                        processing_result["feedback_list"] = feedback_list
                        sentence_analysis["steps"].append(
                            {
                                "step": "get_llm_feedback",
                                "timestamp": datetime.now(UTC),
                                "duration_sec": time.time() - step_start_time,
                                "status": "success",
                                "llm_input": feedback_formatted_input,
                                "llm_output": feedback_response,
                                "output_feedback_count": len(feedback_list),
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error getting LLM feedback",
                            exc_info=True,
                            extra={
                                "response_id": response_id,
                                "sentence_index": sentence_index,
                            },
                        )
                        sentence_analysis["errors"].append(
                            {
                                "step": "get_llm_feedback",
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        sentence_analysis["steps"].append(
                            {
                                "step": "get_llm_feedback",
                                "status": "error",
                                "duration_sec": time.time() - step_start_time,
                            }
                        )
                        processing_result["feedback_list"] = []
                else:
                    logger.warning(
                        f"No refined edits generated, skipping LLM feedback.",
                        extra={
                            "response_id": response_id,
                            "sentence_index": sentence_index,
                        },
                    )
                    sentence_analysis["steps"].append(
                        {
                            "step": "get_llm_feedback",
                            "status": "skipped_no_refined_edits",
                        }
                    )

            elif feedback_system == config.RULE_BASED_SYSTEM:
                step_start_time = time.time()
                feedback_list = []
                rule_based_tasks = [
                    get_rule_based_feedback(
                        pair["sentence"], pair["corrected"], rough_edit, response_id
                    )
                    for rough_edit in rough_edits
                ]
                try:
                    feedback_task_results = await asyncio.gather(
                        *rule_based_tasks, return_exceptions=True
                    )
                    for res_idx, result in enumerate(feedback_task_results):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Error getting rule-based feedback for edit {res_idx}",
                                exc_info=result,
                                extra={
                                    "response_id": response_id,
                                    "sentence_index": sentence_index,
                                },
                            )
                            sentence_analysis["errors"].append(
                                {
                                    "step": "get_rule_based_feedback",
                                    "edit_index": res_idx,
                                    "error": str(result),
                                    "traceback": traceback.format_exc(result),
                                }
                            )
                        else:
                            feedback_list.append(result)

                    processing_result["feedback_list"] = feedback_list
                    sentence_analysis["steps"].append(
                        {
                            "step": "get_rule_based_feedback",
                            "timestamp": datetime.now(UTC),
                            "duration_sec": time.time() - step_start_time,
                            "status": (
                                "success"
                                if not any(
                                    isinstance(r, Exception)
                                    for r in feedback_task_results
                                )
                                else "partial_error"
                            ),
                            "output_feedback_count": len(feedback_list),
                            "raw_results": [
                                str(r) if isinstance(r, Exception) else r
                                for r in feedback_task_results
                            ],
                        }
                    )

                except Exception as e:
                    logger.error(
                        f"Error gathering rule-based feedback tasks",
                        exc_info=True,
                        extra={
                            "response_id": response_id,
                            "sentence_index": sentence_index,
                        },
                    )
                    sentence_analysis["errors"].append(
                        {
                            "step": "gather_rule_based_feedback",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    sentence_analysis["steps"].append(
                        {
                            "step": "get_rule_based_feedback",
                            "status": "error",
                            "duration_sec": time.time() - step_start_time,
                        }
                    )
                    processing_result["feedback_list"] = []
        else:
            logger.debug(
                f"No rough edits found, skipping feedback generation.",
                extra={"response_id": response_id, "sentence_index": sentence_index},
            )
            sentence_analysis["steps"].append(
                {"step": "feedback_generation", "status": "skipped_no_rough_edits"}
            )

        if (
            rough_edits
            and not processing_result["feedback_list"]
            and not any(
                e["step"]
                in [
                    "get_llm_feedback",
                    "get_rule_based_feedback",
                    "gather_rule_based_feedback",
                ]
                for e in sentence_analysis["errors"]
            )
        ):
            logger.warning(
                f"Rough edits found but no feedback generated (and no error logged in feedback step)",
                extra={"response_id": response_id, "sentence_index": sentence_index},
            )

    except Exception as e:
        logger.error(
            f"Unhandled error processing sentence",
            exc_info=True,
            extra={"response_id": response_id, "sentence_index": sentence_index},
        )
        sentence_analysis["errors"].append(
            {
                "step": "unknown_sentence_level_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        processing_result["feedback_list"] = []

    finally:
        sentence_analysis["total_duration_sec"] = time.time() - start_time_sentence
        sentence_analysis["result_feedback_count"] = len(
            processing_result["feedback_list"]
        )

        return processing_result, sentence_analysis
