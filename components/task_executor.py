"""
Task executor component for Jarvis AI Assistant.

Handles execution of various commands and tasks.
"""

import logging
import subprocess
import json
import platform
import datetime
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskExecutor:
    """Handles execution of commands and tasks."""

    def __init__(self, config):
        """
        Initialize the task executor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.reminders = []

    def execute_command(self, command_type, params=None):
        """
        Execute a command based on its type and parameters.
        
        Args:
            command_type (str): Type of command to execute
            params (dict, optional): Parameters for the command
            
        Returns:
            dict: Result of the command execution
        """
        if params is None:
            params = {}
            
        try:
            if command_type == "get_time":
                return self.get_current_time()
            elif command_type == "get_system_info":
                return self.get_system_info()
            elif command_type == "set_reminder":
                return self.set_reminder(params.get("text"), params.get("time"))
            elif command_type == "list_reminders":
                return self.list_reminders()
            elif command_type == "open_application":
                return self.open_application(params.get("app_name"))
            elif command_type == "execute_system_command":
                return self.execute_system_command(params.get("command"), params.get("return_output", True))
            else:
                logger.warning(f"Unknown command type: {command_type}")
                return {"success": False, "error": f"Unknown command type: {command_type}"}
        except Exception as e:
            logger.exception(f"Error executing command {command_type}")
            return {"success": False, "error": str(e)}

    def get_current_time(self):
        """
        Get the current time.
        
        Returns:
            dict: Current time information
        """
        now = datetime.datetime.now()
        return {
            "success": True,
            "time": {
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "formatted": now.strftime("%H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "day_of_week": now.strftime("%A"),
                "timezone": datetime.datetime.now(datetime.timezone.utc).astimezone().tzname()
            }
        }

    def get_system_info(self):
        """
        Get system information.
        
        Returns:
            dict: System information
        """
        system_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }
        
        return {
            "success": True,
            "system_info": system_info
        }

    def set_reminder(self, text, time_str=None):
        """
        Set a reminder with text and optional time.
        
        Args:
            text (str): The reminder text
            time_str (str, optional): Time string for the reminder
            
        Returns:
            dict: Result of setting the reminder
        """
        if not text:
            return {"success": False, "error": "Reminder text is required"}
            
        try:
            reminder_time = None
            if time_str:
                # Simple time parsing
                now = datetime.datetime.now()
                if ":" in time_str:
                    # Format like "14:30"
                    hour, minute = map(int, time_str.split(":"))
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    # If time is in the past, schedule it for tomorrow
                    if reminder_time < now:
                        reminder_time += datetime.timedelta(days=1)
                else:
                    # Try to parse as hour
                    try:
                        hour = int(time_str)
                        reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                        # If time is in the past, schedule it for tomorrow
                        if reminder_time < now:
                            reminder_time += datetime.timedelta(days=1)
                    except ValueError:
                        pass
            
            reminder = {
                "id": len(self.reminders) + 1,
                "text": text,
                "time": reminder_time.isoformat() if reminder_time else None,
                "created_at": datetime.datetime.now().isoformat()
            }
            
            self.reminders.append(reminder)
            
            return {
                "success": True,
                "reminder": reminder
            }
        except Exception as e:
            logger.exception("Error setting reminder")
            return {
                "success": False,
                "error": str(e)
            }

    def list_reminders(self):
        """
        List all reminders.
        
        Returns:
            dict: List of reminders
        """
        return {
            "success": True,
            "reminders": self.reminders
        }

    def open_application(self, app_name):
        """
        Open an application.
        
        Args:
            app_name (str): Name of the application to open
            
        Returns:
            dict: Result of opening the application
        """
        if not app_name:
            return {"success": False, "error": "Application name is required"}
            
        try:
            # This is a simplified implementation
            # In a real implementation, this should handle platform-specific details
            return {
                "success": False,
                "error": "Opening applications is not supported in this environment"
            }
        except Exception as e:
            logger.exception(f"Error opening application: {app_name}")
            return {
                "success": False,
                "error": str(e)
            }

    def execute_system_command(self, command, return_output=True):
        """
        Execute a system command.
        
        Args:
            command (str): Command to execute
            return_output (bool): Whether to return the command output
            
        Returns:
            dict: Result of executing the command
        """
        if not command:
            return {"success": False, "error": "Command is required"}
            
        try:
            # This is a simplified implementation
            # In a real implementation, this should have proper security measures
            return {
                "success": False,
                "error": "Executing system commands is not supported in this environment"
            }
        except Exception as e:
            logger.exception(f"Error executing command: {command}")
            return {
                "success": False,
                "error": str(e)
            }

    def cleanup(self):
        """Clean up resources."""
        # Currently no resources to clean up
        pass