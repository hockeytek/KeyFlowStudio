"""Frame range calculation and state management utilities"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameRangeState:
    """Represents frame range state and calculated updates."""
    start_frame: int      # 0-based
    end_frame: int        # 0-based inclusive, -1 = to end
    frame_count: int      # 0 = all, N = count
    
    # Fields that should be updated
    updated_start_frame: Optional[int] = None
    updated_end_frame: Optional[int] = None
    updated_frame_count: Optional[int] = None


class FrameRangeController:
    """Manages synchronization of frame range controls (From, To, Count).
    
    Priority logic:
    - If count > 0: start_frame + count → end_frame
    - If count == 0: start_frame + end_frame → count
    - When any field changes, calculates what needs to update
    """
    
    @staticmethod
    def on_start_frame_changed(
        new_start: int,
        current_end: int,
        current_count: int
    ) -> FrameRangeState:
        """Handle start frame change. Recalculate end if count is set.
        
        Args:
            new_start: New start frame (0-based)
            current_end: Current end frame (0-based inclusive, -1 = to end)
            current_count: Current count (0 = all)
        
        Returns:
            FrameRangeState with updates needed
        """
        state = FrameRangeState(new_start, current_end, current_count)
        
        # If count is explicitly set, recalculate end frame
        if current_count > 0:
            state.updated_end_frame = new_start + current_count - 1
        
        return state
    
    @staticmethod
    def on_end_frame_changed(
        current_start: int,
        new_end: int,
        current_count: int
    ) -> FrameRangeState:
        """Handle end frame change. Always recalculate count.
        
        Args:
            current_start: Current start frame (0-based)
            new_end: New end frame (0-based inclusive, -1 = to end)
            current_count: Current count (0 = all)
        
        Returns:
            FrameRangeState with updates needed
        """
        state = FrameRangeState(current_start, new_end, current_count)
        
        # Always recalculate count when end changes
        if new_end >= 0:
            state.updated_frame_count = new_end - current_start + 1
        else:
            state.updated_frame_count = 0
        
        return state
    
    @staticmethod
    def on_count_changed(
        current_start: int,
        current_end: int,
        new_count: int
    ) -> FrameRangeState:
        """Handle count change. Recalculate end frame.
        
        Args:
            current_start: Current start frame (0-based)
            current_end: Current end frame (0-based inclusive, -1 = to end)
            new_count: New count (0 = all)
        
        Returns:
            FrameRangeState with updates needed
        """
        state = FrameRangeState(current_start, current_end, new_count)
        
        # Recalculate end frame based on count
        if new_count > 0:
            state.updated_end_frame = current_start + new_count - 1
        else:
            state.updated_end_frame = -1
        
        return state


def calculate_frame_count(start_frame: int, end_frame: int) -> int:
    """Calculate frame count from start and end frames.
    
    Args:
        start_frame: 0-based start frame
        end_frame: 0-based inclusive end frame (-1 means "to the end")
    
    Returns:
        Calculated frame count, or 0 if range is invalid
    
    Example:
        calculate_frame_count(0, 9) -> 10
        calculate_frame_count(5, 15) -> 11
    """
    if end_frame < 0 or end_frame < start_frame:
        return 0
    return end_frame - start_frame + 1


def calculate_end_frame(start_frame: int, frame_count: int) -> int:
    """Calculate end frame from start frame and frame count.
    
    Args:
        start_frame: 0-based start frame
        frame_count: Number of frames to process (0 means "all remaining")
    
    Returns:
        0-based inclusive end frame, or -1 if count is 0 or invalid
    
    Example:
        calculate_end_frame(0, 10) -> 9
        calculate_end_frame(5, 11) -> 15
        calculate_end_frame(0, 0) -> -1
    """
    if frame_count <= 0:
        return -1
    return start_frame + frame_count - 1
