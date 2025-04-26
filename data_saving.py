import boto3
import json
import logging
import os
from datetime import datetime, UTC
from errant.edit import Edit as ErrantEdit

import config

logger = logging.getLogger(__name__)

save_to_s3 = config.SAVE_TO_S3
if save_to_s3:
    s3_client = None
    S3_BUCKET_NAME = config.S3_BUCKET_NAME

def get_s3_client():
    """Initializes and returns the S3 client."""
    global s3_client
    if s3_client is None:
        try:
            s3_client = boto3.client("s3", region_name=config.AWS_REGION)
            logger.info("S3 client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}", exc_info=True)
            s3_client = None
    return s3_client


def default_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ErrantEdit):
        return {
            "o_start": obj.o_start,
            "o_end": obj.o_end,
            "o_str": obj.o_str,
            "c_start": obj.c_start,
            "c_end": obj.c_end,
            "c_str": obj.c_str,
            "type": obj.type,
        }
    try:
        return str(obj)
    except Exception:
        return f"<Object not serializable: {type(obj)}>"

async def save_request_analysis_data(data: dict, response_id: str):
    """
    Saves the provided data dictionary to S3 as a JSON file, or locally if not in AWS.
    Organizes files by date and uses response_id as the filename.
    """    
    now = datetime.now(UTC)
    file_path = f"{config.OUTPUT_DIR}/year={now.year}/month={now.strftime('%m')}/day={now.strftime('%d')}/{response_id}.json"
    json_data = json.dumps(data, indent=2, default=default_serializer)

    if save_to_s3:
        client = get_s3_client()
        if not client or not S3_BUCKET_NAME:
            logger.error(
                "S3 client not initialized or S3_BUCKET_NAME not set. Cannot save data.",
                extra={"response_id": response_id},
            )
            return
        try:
            client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=file_path,
                Body=json_data.encode("utf-8"),
                ContentType="application/json",
            )
            logger.debug(
                f"Successfully saved analysis data to S3",
                extra={
                    "response_id": response_id,
                    "s3_bucket": S3_BUCKET_NAME,
                    "s3_key": file_path,
                },
            )
        except Exception as e:
            logger.error(
                f"Error saving analysis data to S3 for response_id {response_id}: {e}",
                extra={"response_id": response_id, "s3_bucket": S3_BUCKET_NAME},
                exc_info=True,
            )
    else:
        # Save locally
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_data)
            logger.debug(
                f"Successfully saved analysis data locally",
                extra={"response_id": response_id, "local_path": file_path},
            )
        except Exception as e:
            logger.error(
                f"Error saving analysis data locally for response_id {response_id}: {e}",
                extra={"response_id": response_id, "local_path": file_path},
                exc_info=True,
            )
