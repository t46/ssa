"""Terminal output formatter for cleaner console display."""

import sys
from typing import Optional, Union
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Types of messages for formatting."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    AGENT = "agent"
    SYSTEM = "system"
    RESULT = "result"
    PROGRESS = "progress"
    SECTION = "section"
    SUBSECTION = "subsection"


class TerminalFormatter:
    """Formatter for clean terminal output with colors and structure."""
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    # Icons for different message types
    ICONS = {
        MessageType.INFO: "ℹ️ ",
        MessageType.SUCCESS: "✅",
        MessageType.WARNING: "⚠️ ",
        MessageType.ERROR: "❌",
        MessageType.AGENT: "🤖",
        MessageType.SYSTEM: "⚙️ ",
        MessageType.RESULT: "📊",
        MessageType.PROGRESS: "⏳",
        MessageType.SECTION: "📌",
        MessageType.SUBSECTION: "📍",
    }
    
    def __init__(self, use_colors: bool = True, show_timestamp: bool = False):
        """Initialize formatter.
        
        Args:
            use_colors: Whether to use ANSI colors in output
            show_timestamp: Whether to include timestamps
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.show_timestamp = show_timestamp
        
    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        if not self.show_timestamp:
            return ""
        return f"[{datetime.now().strftime('%H:%M:%S')}] "
    
    def print(self, message: str, msg_type: MessageType = MessageType.INFO, indent: int = 0):
        """Print formatted message.
        
        Args:
            message: The message to print
            msg_type: Type of message for formatting
            indent: Number of spaces to indent
        """
        timestamp = self._get_timestamp()
        icon = self.ICONS.get(msg_type, "")
        indent_str = " " * indent
        
        # Color mapping
        color_map = {
            MessageType.INFO: "blue",
            MessageType.SUCCESS: "green",
            MessageType.WARNING: "yellow",
            MessageType.ERROR: "red",
            MessageType.AGENT: "cyan",
            MessageType.SYSTEM: "dim",
            MessageType.RESULT: "magenta",
            MessageType.PROGRESS: "magenta",
            MessageType.SECTION: "bold",
            MessageType.SUBSECTION: "white",
        }
        
        color = color_map.get(msg_type, "white")
        
        # Format based on message type
        if msg_type == MessageType.SECTION:
            separator = "=" * 80
            print(f"\n{self._color(separator, color)}")
            print(f"{timestamp}{indent_str}{icon} {self._color(message.upper(), color)}")
            print(f"{self._color(separator, color)}\n")
        elif msg_type == MessageType.SUBSECTION:
            separator = "-" * 60
            print(f"\n{self._color(separator, 'dim')}")
            print(f"{timestamp}{indent_str}{icon} {self._color(message, color)}")
            print(f"{self._color(separator, 'dim')}")
        else:
            formatted_msg = f"{timestamp}{indent_str}{icon} {message}"
            print(self._color(formatted_msg, color))
    
    def print_agent_message(self, message: str, truncate: bool = True, max_length: int = 200):
        """Print agent messages with special formatting.
        
        Args:
            message: Agent message to print
            truncate: Whether to truncate long messages
            max_length: Maximum length before truncation
        """
        # Clean up the message
        cleaned = message.strip()
        
        # Skip empty messages
        if not cleaned:
            return
        
        # Truncate if needed
        if truncate and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        # Special handling for common agent patterns
        if "Error" in cleaned or "error" in cleaned:
            self.print(cleaned, MessageType.ERROR, indent=2)
        elif "Warning" in cleaned or "warning" in cleaned:
            self.print(cleaned, MessageType.WARNING, indent=2)
        elif any(word in cleaned.lower() for word in ["complete", "success", "done", "finished"]):
            self.print(cleaned, MessageType.SUCCESS, indent=2)
        else:
            self.print(cleaned, MessageType.AGENT, indent=2)

    def print_system_message(self, message: str, truncate: bool = True, max_length: int = 150):
        """Print system messages with special formatting.
        
        Args:
            message: System message to print
            truncate: Whether to truncate long messages
            max_length: Maximum length before truncation
        """
        # Clean up the message
        cleaned = message.strip()
        
        # Skip empty messages
        if not cleaned:
            return
        
        # Truncate if needed
        if truncate and len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        self.print(cleaned, MessageType.SYSTEM, indent=1)

    def print_result_message(self, message: str, details: dict = None):
        """Print result messages with special formatting.
        
        Args:
            message: Result message to print
            details: Optional details dict to include
        """
        # Clean up the message
        cleaned = message.strip()
        
        # Skip empty messages
        if not cleaned:
            return
        
        self.print(cleaned, MessageType.RESULT, indent=1)
        
        # Print details if provided
        if details:
            for key, value in details.items():
                detail_msg = f"{key}: {value}"
                self.print(detail_msg, MessageType.INFO, indent=3)
    
    def print_progress(self, current: int, total: int, task: str = "Processing"):
        """Print progress bar.
        
        Args:
            current: Current item number
            total: Total items
            task: Description of task
        """
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        progress_msg = f"{task}: [{bar}] {percentage:.1f}% ({current}/{total})"
        print(f"\r{self._color(progress_msg, 'magenta')}", end="", flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def clear_line(self):
        """Clear the current line."""
        print("\r" + " " * 80 + "\r", end="", flush=True)


# Global formatter instance
formatter = TerminalFormatter()