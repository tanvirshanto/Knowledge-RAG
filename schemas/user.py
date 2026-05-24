from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    role: Literal["user", "maintainer"] = "user"


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    created_at: str


class UserUpdate(BaseModel):
    role: Optional[Literal["user", "maintainer"]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)

    @model_validator(mode="after")
    def check_at_least_one(self):
        if self.role is None and self.is_active is None and self.password is None:
            raise ValueError("At least one field must be provided for update")
        return self
