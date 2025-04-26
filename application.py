from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager
import logging
import spacy
import uuid
from datetime import datetime, UTC
import traceback

import config
from logging_config import setup_logging
from api_models import userRequest, feedbackResponse
from data_saving import save_request_analysis_data
from processing import process_sentence_pair
from error_correction import get_corrections, load_correction_llm
from feedback_generation import (
    load_feedback_llm,
)
from edit_extraction import (
    load_errant,
    load_edit_llm,
)

logger = setup_logging(
    log_level=logging.INFO, enable_file_logging=config.ENABLE_FILE_LOG
)

# Global variables
correction_llm = None
explanation_llm = None
feedback_llm = None
errant_annotator = None
nlp = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load resources once at startup and clean up at shutdown."""
    global nlp, errant_annotator, correction_llm, explanation_llm, feedback_llm
    nlp = spacy.load("en_core_web_sm")
    errant_annotator = load_errant()
    correction_llm = load_correction_llm()
    explanation_llm = load_edit_llm()
    feedback_llm = load_feedback_llm()
    yield

    # Clean up the models and release the resources
    del nlp
    del errant_annotator
    del correction_llm
    del explanation_llm
    del feedback_llm


application = FastAPI(lifespan=lifespan)

# TODO: Determine how to use CORS for the AWS deployment
# CORS settings
application.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@application.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if the resources are loaded
    resources = {
        "spacy": nlp,
        "errant_annotator": errant_annotator,
        "correction_llm": correction_llm,
        "explanation_llm": explanation_llm,
        "feedback_llm": feedback_llm,
    }
    for name, resource in resources.items():
        if resource is None:
            logger.error(f"Health check result: {name} is not loaded")
            return {"status": "error", "message": f"{name} is not loaded"}
    logger.info("Health check result: All resources are loaded")

    return {"status": "ok"}


@application.post("/grammar_feedback/", response_model=feedbackResponse)
async def grammar_feedback(request: userRequest):
    response_id = str(uuid.uuid4())
    # Initialize the dictionary to store all data for analysis
    # This will be saved to S3 at the end of processing
    # NOTE: This approach is responsible for about half the length of the files...
    # However, I really wanted a reliable step-by-step analysis of the processing
    analysis_data = {
        "request_id": response_id,
        "user_id": request.user_id,
        "draft_number": request.draft_number,
        "request_timestamp": datetime.now(UTC),
        "request_text": request.text,
        "system_choice": request.system_choice,
        "processing_steps": [],
        "errors": [],
        "final_api_response": None,
        "metadata": {},
    }

    try:
        # Input Validation
        if not request.text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        if len(request.text) > config.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text length exceeds limit of {config.MAX_TEXT_LENGTH} characters.",
            )

        logger.info(
            "Processing request",
            extra={
                "response_id": response_id,
                "user_id": request.user_id,
                "draft_number": request.draft_number,
                "text_length": len(request.text),
                "system_choice": request.system_choice,
            },
        )

        doc = nlp(request.text)
        spacy_sentences = list(doc.sents)
        sentence_texts = [sent.text for sent in spacy_sentences]
        sentence_map = {sent.text: sent for sent in spacy_sentences}
        analysis_data["processing_steps"].append(
            {
                "step": "sentence_splitting",
                "timestamp": datetime.now(UTC),
                "input_length": len(request.text),
                "output_sentence_count": len(sentence_texts),
            }
        )

        # GEC Correction Step
        corrected_sentences, gec_formatted_input, gec_response = [], None, None
        try:
            corrected_sentences, gec_formatted_input, gec_response = (
                await get_corrections(sentences=sentence_texts, response_id=response_id)
            )
            analysis_data["processing_steps"].append(
                {
                    "step": "gec_correction",
                    "timestamp": datetime.now(UTC),
                    "status": "success",
                    "llm_input": gec_formatted_input,
                    "llm_output": gec_response,
                    "corrected_sentences": corrected_sentences,
                }
            )
        except Exception as e:
            logger.error(
                f"Error correcting text for response_id: {response_id}: {e}",
                exc_info=True,
                extra={"response_id": response_id},
            )
            analysis_data["errors"].append(
                {
                    "step": "gec_correction",
                    "timestamp": datetime.now(UTC),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            raise HTTPException(status_code=500, detail="Error correcting text")

        # Prepare Sentence Pairs
        sentence_pairs = (
            [
                {"sentence": s, "corrected": c}
                for s, c in zip(sentence_texts, corrected_sentences)
            ]
            if len(sentence_texts) == len(corrected_sentences)
            else []
        )

        if len(sentence_texts) != len(corrected_sentences):
            logger.error(
                "Mismatch between original and corrected sentence counts",
                extra={"response_id": response_id},
            )
            analysis_data["errors"].append(
                {
                    "step": "sentence_pair_creation",
                    "timestamp": datetime.now(UTC),
                    "error": "Mismatch count",
                    "original_count": len(sentence_texts),
                    "corrected_count": len(corrected_sentences),
                }
            )
            sentence_pairs = []
        if not sentence_pairs:
            logger.warning(
                "No sentence pairs created for feedback generation",
                extra={"response_id": response_id},
            )
            analysis_data["errors"].append(
                {
                    "step": "sentence_pair_creation",
                    "timestamp": datetime.now(UTC),
                    "error": "No sentence pairs created",
                }
            )
            raise HTTPException(
                status_code=400, detail="Error aligning corrected sentences"
            )

        # Feedback System Selection
        if request.system_choice in config.SYSTEMS_LIST:
            feedback_system = request.system_choice
        else:
            feedback_system = config.DEFAULT_SYSTEM
        analysis_data["selected_feedback_system"] = feedback_system
        logger.info(
            f"Selected feedback system: {feedback_system}",
            extra={"response_id": response_id},
        )

        # Process Sentences Concurrently
        sentence_level_results = []
        sentence_analysis_data = []
        if sentence_pairs:
            try:
                sentence_tasks = [
                    process_sentence_pair(
                        pair=pair,
                        response_id=response_id,
                        feedback_system=feedback_system,
                        sentence_index=idx,
                    )
                    for idx, pair in enumerate(sentence_pairs)
                ]
                # Each result should be a tuple: (api_result_for_sentence, analysis_data_for_sentence)
                gathered_results = await asyncio.gather(
                    *sentence_tasks, return_exceptions=True
                )

                for i, result in enumerate(gathered_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Error processing sentence pair {i}",
                            exc_info=result,
                            extra={"response_id": response_id},
                        )
                        analysis_data["errors"].append(
                            {
                                "step": "process_sentence_pair",
                                "sentence_index": i,
                                "timestamp": datetime.now(UTC),
                                "error": str(result),
                                "traceback": traceback.format_exc(result),
                            }
                        )
                        sentence_level_results.append(None)  # Later filtered
                        sentence_analysis_data.append(
                            {
                                "sentence_index": i,
                                "status": "error",
                                "error": str(result),
                            }
                        )
                    else:
                        api_result, analysis_result = result
                        sentence_level_results.append(api_result)
                        sentence_analysis_data.append(analysis_result)

            except Exception as e:
                logger.error(
                    f"Error gathering sentence processing tasks for response_id: {response_id}: {e}",
                    exc_info=True,
                    extra={"response_id": response_id},
                )
                analysis_data["errors"].append(
                    {
                        "step": "gather_sentence_tasks",
                        "timestamp": datetime.now(UTC),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                raise HTTPException(status_code=500, detail="Error generating feedback")

        analysis_data["sentence_processing_results"] = sentence_analysis_data

        # Assemble Final API Response
        api_response = dict(response_id=response_id, feedback_list=[], metadata={})
        for sentence_result in sentence_level_results:
            if sentence_result and "feedback_list" in sentence_result:
                api_response["feedback_list"].extend(sentence_result["feedback_list"])

        # Add Global Offsets and Metadata
        for i, feedback in enumerate(api_response["feedback_list"]):
            # Add external index to each feedback item:
            feedback["index"] = i
            try:
                spacy_sentence = sentence_map[feedback["source"]]
                start_char = spacy_sentence.start_char
                feedback["global_highlight_start"] = (
                    start_char + feedback["highlight_start"]
                )
                feedback["global_highlight_end"] = (
                    start_char + feedback["highlight_end"]
                )
            except (KeyError, TypeError, AttributeError) as e:
                logger.warning(
                    f"Could not calculate global offsets for feedback item {i}",
                    exc_info=True,
                    extra={"response_id": response_id, "feedback_item": feedback},
                )
                feedback["global_highlight_start"] = None
                feedback["global_highlight_end"] = None

        # Populate metadata for API response
        api_response["metadata"]["timestamp"] = datetime.now(UTC)
        api_response["metadata"]["api_version"] = config.API_VERSION
        api_response["metadata"]["system_used"] = feedback_system

        analysis_data["final_api_response"] = api_response
        analysis_data["metadata"] = api_response["metadata"]
        analysis_data["processing_end_timestamp"] = datetime.now(UTC)

        logger.info(
            f"Response generated successfully", extra={"response_id": response_id}
        )
        return api_response

    except Exception as e:
        logger.error(
            f"Unhandled exception in grammar_feedback for response_id: {response_id}: {e}",
            exc_info=True,
            extra={"response_id": response_id},
        )
        analysis_data["errors"].append(
            {
                "step": "unhandled_exception",
                "timestamp": datetime.now(UTC),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

    finally:
        await save_request_analysis_data(analysis_data, response_id)
