import glob
import os
import numpy as np
import pytest
import warnings
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open
from woundcompute import image_analysis as ia
from woundcompute import warnings_handler as warn


def files_path():
    self_path_file = Path(__file__)
    self_path = self_path_file.resolve().parent
    data_path = self_path.joinpath("files").resolve()
    return data_path


def example_path(example_name):
    data_path = files_path()
    example_path = data_path.joinpath(example_name).resolve()
    return example_path


def test_save_bg_shift_warning():
    folder_path = example_path("test_ph1_movie_mini_large_bg_shift")
    sample_name = "test_sample"
    large_shift_frame_ind = np.array([1])
    _, _ = ia.run_all(folder_path)
    output_p = warn.save_bg_shift_warning(folder_path,sample_name,large_shift_frame_ind)
    assert output_p.is_file()


class TestWarningLogger:
    """Test suite for WarningLogger class using pytest features"""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Fixture providing temporary directory for tests"""
        return tmp_path / "test_logs"
    
    @pytest.fixture
    def warning_logger(self, temp_log_dir):
        """Fixture providing WarningLogger instance"""
        return warn.WarningLogger(temp_log_dir)
    
    def test_init_creates_directory(self, temp_log_dir):
        """Test that __init__ creates the log directory"""
        # Ensure directory doesn't exist initially
        assert not temp_log_dir.exists()
        
        # Create logger which should create directory
        logger = warn.WarningLogger(temp_log_dir)
        
        # Verify directory was created
        assert logger.log_dir.exists()
        assert logger.log_dir.name == "gpr_warnings"
    
    def test_init_custom_folder_name(self, temp_log_dir):
        """Test initialization with custom folder name"""
        custom_folder = "custom_warnings"
        logger = warn.WarningLogger(temp_log_dir, folder_name=custom_folder)
        
        assert logger.log_dir.name == custom_folder
        assert logger.log_dir.exists()
    
    def test_context_manager_usage(self, temp_log_dir):
        """Test that WarningLogger works as context manager"""
        original_showwarning = warnings.showwarning
        
        with warn.WarningLogger(temp_log_dir) as logger:
            # Verify warning handler was changed
            assert warnings.showwarning != original_showwarning
            assert warnings.showwarning == logger.custom_warning_handler
        
        # Verify warning handler was restored
        assert warnings.showwarning == original_showwarning
    
    def test_start_stop_logging(self, warning_logger):
        """Test start_logging and stop_logging methods"""
        original_showwarning = warnings.showwarning
        
        warning_logger.start_logging()
        assert warnings.showwarning == warning_logger.custom_warning_handler
        
        warning_logger.stop_logging()
        assert warnings.showwarning == original_showwarning

    
    def test_warning_redirection_integration(self, temp_log_dir, monkeypatch):
        """Integration test: verify warnings are actually redirected"""
        logger = warn.WarningLogger(temp_log_dir)
        
        # Capture the log file content using monkeypatch
        captured_content = []
        
        def mock_open_file(filepath, mode):
            class MockFile:
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
                
                def write(self, content):
                    captured_content.append(content)
            
            return MockFile()
        
        monkeypatch.setattr('builtins.open', mock_open_file)
        
        logger.start_logging()
        
        # Generate a warning
        warnings.warn("Integration test warning", UserWarning)
        
        logger.stop_logging()
        
        # Verify warning was captured
        assert len(captured_content) == 1
        assert "Integration test warning" in captured_content[0]
        assert "UserWarning" in captured_content[0]
    
    def test_multiple_warnings(self, temp_log_dir, monkeypatch):
        """Test that multiple warnings are logged correctly"""
        logger = warn.WarningLogger(temp_log_dir)
        
        captured_writes = []
        
        def mock_open_file(filepath, mode):
            class MockFile:
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
                
                def write(self, content):
                    captured_writes.append(content)
            
            return MockFile()
        
        monkeypatch.setattr('builtins.open', mock_open_file)
        
        logger.start_logging()
        
        # Generate multiple warnings
        for i in range(3):
            warnings.warn(f"Warning {i}", UserWarning)
        
        logger.stop_logging()
        
        # Verify all warnings were captured
        assert len(captured_writes) == 3
        for i in range(3):
            assert f"Warning {i}" in captured_writes[i]
    
    def test_different_warning_categories(self, temp_log_dir, monkeypatch):
        """Test handling of different warning categories"""
        logger = warn.WarningLogger(temp_log_dir)
        
        captured_content = []
        
        def mock_open_file(filepath, mode):
            class MockFile:
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
                
                def write(self, content):
                    captured_content.append(content)
            
            return MockFile()
        
        monkeypatch.setattr('builtins.open', mock_open_file)
        
        logger.start_logging()
        
        # Generate different types of warnings
        warnings.warn("User warning", UserWarning)
        warnings.warn("Deprecation warning", DeprecationWarning)
        
        logger.stop_logging()
        
        # Verify both warning types were handled
        assert any("UserWarning" in content for content in captured_content)
        assert any("DeprecationWarning" in content for content in captured_content)
    
    def test_original_warning_handler_preserved(self, warning_logger):
        """Test that original warning handler is properly preserved and restored"""
        original_handler = warnings.showwarning
        
        # Store the original handler that was captured during init
        assert warning_logger.original_showwarning == original_handler
        
        # Change the warning handler externally
        def dummy_handler(*args, **kwargs):
            pass
        
        warnings.showwarning = dummy_handler
        
        # Create new logger should capture the new dummy handler
        new_logger = warn.WarningLogger(Path("/tmp"))
        assert new_logger.original_showwarning == dummy_handler
        
        # Restore original
        warnings.showwarning = original_handler


# # Parametrized test example (pytest feature)
# @pytest.mark.parametrize("folder_name,file_name,expected_pattern", [
#     ("test_warnings", "errors", "errors_"),
#     ("alerts", "notices", "notices_"),
#     ("", "generic", "generic_"),  # Edge case
# ])
# def test_log_file_patterns(tmp_path, folder_name, file_name, expected_pattern):
#     """Test various file and folder name combinations"""
#     logger = warn.WarningLogger(tmp_path, folder_name=folder_name, file_name=file_name)
    
#     if folder_name:
#         assert logger.log_dir.name == folder_name
#     assert expected_pattern in logger.log_file.name
#     assert logger.log_file.suffix == ".log"


# # Fixture with params (pytest feature)
# @pytest.fixture(params=[UserWarning, DeprecationWarning, FutureWarning])
# def warning_type(request):
#     """Fixture providing different warning types"""
#     return request.param


# def test_various_warning_types(temp_log_dir, warning_type, monkeypatch):
#     """Test that different warning types are handled"""
#     captured = []
    
#     def mock_open_file(filepath, mode):
#         class MockFile:
#             def __enter__(self):
#                 return self
            
#             def __exit__(self, *args):
#                 pass
            
#             def write(self, content):
#                 captured.append(content)
        
#         return MockFile()
    
#     monkeypatch.setattr('builtins.open', mock_open_file)
    
#     logger = warn.WarningLogger(temp_log_dir)
#     logger.start_logging()
    
#     warnings.warn("Test warning", warning_type)
    
#     logger.stop_logging()
    
#     assert len(captured) == 1
#     assert warning_type.__name__ in captured[0]