"""
User schemas for Kaizen enterprise data platform.

This module defines Pydantic models for user-related data structures
including authentication, authorization, and user management.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    """Base user model with common fields."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name of the user")
    is_active: bool = Field(True, description="Whether the user is active")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=100, description="Plain text password")
    is_admin: bool = Field(False, description="Whether the user has admin privileges")


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = Field(None)
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = Field(None)
    is_admin: Optional[bool] = Field(None)


class UserChangePassword(BaseModel):
    """Schema for changing user password."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")


class User(UserBase):
    """
    Full user model with all fields.
    
    This is the main user model used throughout the application.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: str = Field(..., description="Unique user identifier")
    hashed_password: Optional[str] = Field(None, description="Hashed password (internal use)")
    is_admin: bool = Field(False, description="Whether the user has admin privileges")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="User creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # Additional fields for enterprise features
    department: Optional[str] = Field(None, max_length=100, description="User department")
    role: Optional[str] = Field(None, max_length=50, description="User role")
    permissions: Optional[list[str]] = Field(default_factory=list, description="User permissions")
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            bool: True if user has permission
        """
        return self.is_admin or permission in (self.permissions or [])
    
    def is_member_of_department(self, department: str) -> bool:
        """
        Check if user belongs to a specific department.
        
        Args:
            department: Department to check
            
        Returns:
            bool: True if user is in department
        """
        return self.department == department if self.department else False


class UserPublic(BaseModel):
    """Public user model without sensitive information."""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    department: Optional[str]
    role: Optional[str]


class UserInDB(User):
    """User model as stored in database (includes hashed password)."""
    hashed_password: str


# Authentication related schemas

class Token(BaseModel):
    """JWT token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserPublic = Field(..., description="User information")


class TokenData(BaseModel):
    """Token payload data model."""
    user_id: str = Field(..., description="User ID from token")
    username: Optional[str] = Field(None, description="Username from token")
    permissions: Optional[list[str]] = Field(default_factory=list, description="User permissions")


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Login response schema."""
    user: UserPublic = Field(..., description="User information")
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str = Field(..., description="Refresh token")


# User management schemas

class UserList(BaseModel):
    """Schema for paginated user list."""
    users: list[UserPublic] = Field(..., description="List of users")
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class UserStats(BaseModel):
    """User statistics schema."""
    total_users: int = Field(..., description="Total number of users")
    active_users: int = Field(..., description="Number of active users")
    inactive_users: int = Field(..., description="Number of inactive users")
    admin_users: int = Field(..., description="Number of admin users")
    users_created_today: int = Field(..., description="Users created today")
    users_created_this_week: int = Field(..., description="Users created this week")
    users_created_this_month: int = Field(..., description="Users created this month")


# Permission and role schemas

class Permission(BaseModel):
    """Permission model."""
    name: str = Field(..., description="Permission name")
    description: Optional[str] = Field(None, description="Permission description")
    category: Optional[str] = Field(None, description="Permission category")


class Role(BaseModel):
    """Role model."""
    name: str = Field(..., description="Role name")
    description: Optional[str] = Field(None, description="Role description")
    permissions: list[str] = Field(default_factory=list, description="Role permissions")


# Activity and audit schemas

class UserActivity(BaseModel):
    """User activity log schema."""
    id: str = Field(..., description="Activity ID")
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Action performed")
    resource: Optional[str] = Field(None, description="Resource accessed")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Activity timestamp")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")


class UserSession(BaseModel):
    """User session model."""
    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    is_active: bool = Field(True, description="Whether session is active")


# Password reset schemas

class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")


# User preferences and settings

class UserPreferences(BaseModel):
    """User preferences schema."""
    theme: Optional[str] = Field("light", description="UI theme preference")
    language: Optional[str] = Field("en", description="Language preference")
    timezone: Optional[str] = Field("UTC", description="Timezone preference")
    notifications: dict = Field(default_factory=dict, description="Notification preferences")
    dashboard_layout: Optional[dict] = Field(default_factory=dict, description="Dashboard layout preferences")


class UserProfile(BaseModel):
    """Extended user profile schema."""
    model_config = ConfigDict(from_attributes=True)
    
    user: UserPublic = Field(..., description="User information")
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="User preferences")
    stats: dict = Field(default_factory=dict, description="User statistics")
    recent_activity: list[UserActivity] = Field(default_factory=list, description="Recent user activity")