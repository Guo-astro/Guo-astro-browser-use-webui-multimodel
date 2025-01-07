"""Main entry point for the Browser Use WebUI application."""

import argparse
import logging
from typing import Optional
from ui import create_application_ui

def _configure_logging() -> None:
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def _parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Launch the Browser Use WebUI application"
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        default="127.0.0.1",
        help="IP address to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7788,
        help="Port number to listen on (default: 7788)"
    )
    
    parser.add_argument(
        "--theme",
        type=str,
        default="Ocean",
        choices=["Default", "Soft", "Monochrome", "Glass", "Origin", "Citrus", "Ocean"],
        help="Visual theme for the application interface (default: Ocean)"
    )
    
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Enable dark mode for the application interface"
    )
    
    return parser.parse_args()

def _launch_application(ip: str, port: int, theme: str) -> None:
    """
    Launch the application with the specified configuration.
    
    Args:
        ip: IP address to bind to
        port: Port number to listen on
        theme: Visual theme to use
    """
    try:
        logging.info("Starting Browser Use WebUI application")
        interface = create_application_ui(theme_name=theme)
        interface.launch(
            server_name=ip,
            server_port=port
        )
    except Exception as e:
        logging.error(f"Failed to launch application: {str(e)}")
        raise

def main() -> None:
    """Main entry point for the Browser Use WebUI application."""
    _configure_logging()
    args = _parse_arguments()
    _launch_application(args.ip, args.port, args.theme)

if __name__ == '__main__':
    main()
