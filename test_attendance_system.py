import pytest
from unittest.mock import MagicMock, patch
import os
import cv2
import numpy as np
from attendance_system import load_known_faces, identify_face_in_frame

# Helper fixture to create a dummy image for mocking frame
# --- Fixtures ---
@pytest.fixture
def dummy_frame():
    """Returns a simple 100x100 black image (numpy array) for testing DeepFace calls."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

# --- Test load_known_faces function ---
@patch('os.path.exists', return_value=True)
@patch('os.listdir')
# The FIX: Mock os.path.isdir to return True only for the expected student folders,
# preventing the function from calling os.listdir on 'IgnoreFile.txt' and causing StopIteration.
@patch('os.path.isdir', side_effect=lambda x: 'Alice' in x or 'Bob' in x) 
def test_load_known_faces_success(mock_isdir, mock_listdir, mock_exists):
    """Tests if known faces are loaded correctly based on mocked file system."""
    
    # Setup the sequence of return values for os.listdir calls:
    mock_listdir.side_effect = [
        ['Alice', 'Bob', 'IgnoreFile.txt'],    # 1. First call (Base Directory) go back to check
        ['alice1.jpg', 'alice_ignore.txt'],    # 2. Second call (Alice's Folder)
        ['bob1.png', 'bob2.jpeg']              # 3. Third call (Bob's Folder)
    ]
    
    base_dir = "test_dir"
    known_faces = load_known_faces(base_dir)
    
    expected_faces = [
        ('Alice', os.path.join(base_dir, 'Alice', 'alice1.jpg')),
        ('Bob', os.path.join(base_dir, 'Bob', 'bob1.png')),
        ('Bob', os.path.join(base_dir, 'Bob', 'bob2.jpeg')),
    ]
    
    assert known_faces == expected_faces
    assert len(known_faces) == 3

def test_load_known_faces_no_dir():
    """Tests behavior when the base directory does not exist."""
    # os.path.exists will return False by default (or can be explicitly mocked)
    known_faces = load_known_faces("non_existent_path")
    assert known_faces == []

# --- Test identify_face_in_frame function ---

@patch('attendance_system.DeepFace.verify')
def test_identify_face_in_frame_match_found(mock_verify, dummy_frame):
    """Tests if a match is correctly identified and returned."""
    
    # Mock DeepFace.verify to simulate a match on the first known face (Alice)
    mock_verify.side_effect = [
        {"verified": True, "distance": 0.1},  # Match!
        {"verified": False, "distance": 0.9} 
    ]
    
    known_faces = [
        ("Alice", "path/to/alice.jpg"),  #go back to check
        ("Bob", "path/to/bob.jpg")
    ]
    
    result = identify_face_in_frame(dummy_frame, known_faces)
    
    # Assert it returns the name of the matched student
    assert result == "Alice"
    # Assert it only called verify once (because the function should 'break' on the match)
    assert mock_verify.call_count == 1
    
@patch('attendance_system.DeepFace.verify')
def test_identify_face_in_frame_no_match(mock_verify, dummy_frame):
    """Tests when the frame does not match any known face."""
    
    # Mock DeepFace.verify to simulate no matches for both
    mock_verify.return_value = {"verified": False, "distance": 0.8}
    
    known_faces = [
        ("Alice", "path/to/alice.jpg"),
        ("Bob", "path/to/bob.jpg")   #
    ]
    
    result = identify_face_in_frame(dummy_frame, known_faces)
    
    # Assert it returns None
    assert result is None
    # Assert it checked both known faces
    assert mock_verify.call_count == 2
    
@patch('attendance_system.DeepFace.verify')
def test_identify_face_in_frame_deepface_error(mock_verify, dummy_frame):
    """Tests error handling (e.g., DeepFace fails to detect a face)."""
    
    # Simulate a DeepFace Exception on the first check, but a match on the second
    mock_verify.side_effect = [
        Exception("No face detected in image"), # Alice fails
        {"verified": True, "distance": 0.1}     # Bob succeeds
    ]
    
    known_faces = [
        ("Alice", "path/to/alice.jpg"),
        ("Bob", "path/to/bob.jpg")
    ]
    
    result = identify_face_in_frame(dummy_frame, known_faces)
    
    # The function should skip the error and continue, finding 'Bob'
    assert result == "Bob"
    assert mock_verify.call_count == 2