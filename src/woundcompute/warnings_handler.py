
import warnings
import os
from datetime import datetime
from typing import Optional
from pathlib import Path
import numpy as np

def save_bg_shift_warning(warning_folder_path: Path, sample_name: str, large_shift_frame_ind: np.ndarray):
    """
    Save a warning text file about potential large background shifts between frames.
    
    Parameters:
        warning_folder_path (Path): Path to the folder where the warning file will be saved
        sample_name (str): Sample name used for the output filename
        large_shift_frame_ind (np.ndarray): Array of frame indices where potential large shifts occur
                                           (1-based indexing, shifts occur between frame_ind and frame_ind+1)
    """
    # Create the output filename
    output_filename = f"{sample_name}_bg_shift_warning.txt"
    output_path = warning_folder_path / output_filename
    
    # Prepare the warning message
    warning_lines = [
        f"Warning: Potential large background shifts detected for sample {sample_name}",
        "------------------------------------------------------------",
        "",
        "The following frame transitions may have large background shifts:",
        "(Note: Frame numbers start from 1)",
        ""
    ]
    
    # Add each shift location to the message
    for frame_ind in large_shift_frame_ind:
        warning_lines.append(f"- Between frame {frame_ind} and frame {frame_ind + 1}")
    
    # Add some additional notes
    warning_lines.extend([
        "",
        "Note: These are potential shifts that may need visual inspection."
    ])
    
    # Combine all lines with newline characters
    warning_text = "\n".join(warning_lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(warning_text)
    
    print(f"Warning file saved to: {output_path}")
    return output_path


class WarningLogger:
    def __init__(self, log_dir:Path, folder_name:str="gpr_warnings", file_name:str="warnings"):
        log_dir = log_dir.joinpath(folder_name).resolve()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{file_name}_{timestamp}.log")

        # Save original warning handler
        self.original_showwarning = warnings.showwarning
        
    def __enter__(self):
        """Context manager entry"""
        self.start_logging()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_logging()
        
    def custom_warning_handler(self, message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that writes to file"""
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} - {category.__name__}: {message} (File: {filename}, Line: {lineno})\n")
        
        # Also show in terminal if you want (optional)
        # self.original_showwarning(message, category, filename, lineno, file, line)
    
    def start_logging(self):
        """Start redirecting warnings to file"""
        warnings.showwarning = self.custom_warning_handler
        
    def stop_logging(self):
        """Restore original warning behavior"""
        warnings.showwarning = self.original_showwarning