from pydantic import BaseModel
import types
from typing import List, Optional, Protocol, Literal, Callable, Dict, Any, Tuple, ClassVar


class SFTData(BaseModel):
  initial_prompt: str
  initial_response: str
  critique_response: str
  revision_response: str
  
class SFTDataset(BaseModel):
  sft_data: List[SFTData]

# Aligned with the HuggingFace cai-conversation-harmless dataset here: https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless
class DPOData(BaseModel):
  # - `initial_prompt`: `(str)` The initial prompt that the user sees.
  # - `initial_response`: `(str)` The initial response that the user sees.
  # - `critique_prompt`: `(str)` The prompt that incorporates the relevant principle and requests a critique from the model
  # - `critique_response`: `(str)` The response that the model provides to the critic prompt
  # - `revision_prompt`: `(str)` The prompt that incorporates the relevant principle and requests a revision from the model
  # - `revision_response`: `(str)` The response that the model provides to the revision prompt
  # - `prompt`: `(str)` Same as the initial prompt
  # - `messages`: `(List[str])` TBD
  # - `chosen`: `(List[str])` The revised and correct response
  # - `rejected`: `(List[str])` The incorrect response
  # - `principle`: `(str)` The principle that the model is being evaluated against
  initial_prompt: str
  initial_response: str
  critique_prompt: str
  critique_response: str
  revision_prompt: str
  revision_response: str
  
class DPODataset(BaseModel):
  dpo_data: List[DPOData]
  
