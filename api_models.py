from pydantic import BaseModel
from typing import List


class userRequest(BaseModel):
    user_id: str
    system_choice: str
    draft_number: int
    text: str


class feedbackComment(BaseModel):
    source: str
    corrected: str
    highlight_start: int
    highlight_end: int
    highlight_text: str
    error_tag: str
    feedback_explanation: str
    feedback_suggestion: str


class responseFeedbackComment(feedbackComment):
    index: int
    global_highlight_start: int
    global_highlight_end: int


class feedbackResponse(BaseModel):
    response_id: str
    feedback_list: List[responseFeedbackComment]
    metadata: dict
