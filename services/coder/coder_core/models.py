from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CodeGenRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    files: Optional[List[str]] = None
    constraints: Optional[str] = None


class CodeFile(BaseModel):
    path: str
    content: str


class CodeGenResponse(BaseModel):
    files: List[CodeFile]
