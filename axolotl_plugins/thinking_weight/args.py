from typing import Optional

from pydantic import BaseModel, Field


class ThinkingWeightArgs(BaseModel):
    thinking_token_weight: Optional[float] = Field(
        default=1.0, description="Weight for tokens inside the thinking window."
    )
    answer_token_weight: Optional[float] = Field(
        default=1.5, description="Weight for tokens outside the thinking window."
    )
    use_thinking_weight: Optional[bool] = Field(
        default=False, description="Whether to use thinking weight."
    )
    think_start_token: str = Field(
        default="<think>", description="Tag for the start of the thinking window."
    )
    think_end_token: str = Field(
        default="</think>", description="Tag for the end of the thinking window."
    )
