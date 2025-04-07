import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import time
from config import *
from collections import deque
import threading
from queue import Queue, Empty, Full

from prompt_utils import *

# Initialize Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
camera = None
model = None
persistent_chat = None  # Persistent chat context
is_initialized = False  # Add initialization flag
is_recording = False  # Recording state flag
recorded_frames = []  # Store recorded frames

def initialize_model():
    """Initialize model and persistent chat context"""
    global model
    if st.session_state.is_initialized:  # If already initialized, return directly
        print("Model already initialized, skipping...")
        return model
        
    print("Initializing model...")  # Debug info
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Use new initialization function
    st.session_state.persistent_chat = init_persistent_chat(model)
    
    print("Model and base context initialization complete")  # Debug info
    st.session_state.is_initialized = True  # Set initialization flag
    return model

def process_recorded_frames():
    """Process recorded frames and return recognition results"""
    print("recorded_frames: ", len(st.session_state.recorded_frames))
    if not st.session_state.recorded_frames or len(st.session_state.recorded_frames) < 6:
        return "Not enough frames, cannot recognize", None
    
    try:
        # Use the latest frames for processing
        latest_frames = st.session_state.recorded_frames
        
        # Resize frames for horizontal display of 6 frames
        resized_frames = []
        for frame in latest_frames:
            width = frame.shape[1] // 6  # Adjust width to 1/6 of original
            height = frame.shape[0]
            resized = cv2.resize(frame, (width, height))
            resized_frames.append(resized)
        
        # Check if persistent_chat exists
        if st.session_state.persistent_chat is None:
            print("persistent_chat not initialized, reinitializing...")
            initialize_model()
        
        # Call processing function from prompt_utils.py
        result = process_images_with_api(latest_frames, st.session_state.persistent_chat)
        
        # Return results and resized frames
        return result, resized_frames
    except Exception as e:
        print(f"Error processing frames: {str(e)}")
        return "UNKNOWN", None

def initialize_camera():
    """Initialize camera and model"""
    global camera, model
    if camera is None:
        print("Initializing camera...")  # Debug info
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        print("Camera initialization complete")  # Debug info
        
        # Initialize model only if not already initialized
        if not st.session_state.is_initialized:
            model = initialize_model()
            print("Model initialization complete")  # Debug info

def cleanup():
    """Clean up resources"""
    global camera
    print("Starting resource cleanup...")  # Debug info
    st.session_state.recording_active = False
    st.session_state.recorded_frames = []
    if camera:
        camera.release()
    camera = None
    st.session_state.is_initialized = False  # Reset initialization flag
    st.session_state.camera_initialized = False  # Reset camera initialization flag
    print("Resource cleanup complete")  # Debug info

# Initialize session state
if 'recognized_words' not in st.session_state:
    st.session_state.recognized_words = []
if 'recognition_results' not in st.session_state:
    st.session_state.recognition_results = deque(maxlen=5)  # Store recognition results
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized = False
if 'recorded_frames' not in st.session_state:
    st.session_state.recorded_frames = []
if 'persistent_chat' not in st.session_state:
    st.session_state.persistent_chat = None

def main():
    global camera
    
    st.title("Sign Language Recognition System")
    st.write("Record sign language gestures and recognize them")

    # Create sidebar for control buttons
    with st.sidebar:
        st.subheader("Control Panel")
        
        # Start/Stop recording buttons
        if st.button("Start Recording", key="start_recording_button", type="primary"):
                print("Start recording button clicked")  # Debug info
                st.session_state.recording_active = True
                st.session_state.recorded_frames = []  # Clear previous recording
                st.success("Recording started! Please make sign language gestures.")
        if st.button("Stop Recording and Recognize", key="stop_recording_button", type="secondary"):
                print("Stop recording button clicked")  # Debug info
                st.session_state.recording_active = False
                st.info("Recording stopped, processing...")
                
                # Process recorded frames
                result, processed_frames = process_recorded_frames()
                st.session_state.processing_result = result
                st.session_state.processed_frames = processed_frames
                
                # Save results
                if result != "UNKNOWN" and result != "Not enough frames, cannot recognize":
                    st.session_state.recognition_results.append(result)
                    # Keep results list no more than 5
                    while len(st.session_state.recognition_results) > 5:
                        st.session_state.recognition_results.popleft()
                
                st.success(f"Recognition complete! Result: {result}")
        
        st.divider()
        
        if st.button("Clear Results", key="clear_results_button"):
            print("Clear results button clicked")  # Debug info
            st.session_state.recognition_results.clear()
            st.session_state.processing_result = None
            st.session_state.processed_frames = None
            st.success("Results cleared!")
        
        st.divider()
        
        if st.button("Exit", key="exit_button", type="secondary"):
            print("Exit button clicked")  # Debug info
            cleanup()
            st.stop()

    # Create two-column layout
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Real-time Camera Feed")
        camera_placeholder = st.empty()
        
        # Display recording status
        if st.session_state.recording_active:
            st.warning("Recording in progress...")
        else:
            st.info("Not recording")

    with col2:
        st.subheader("Processed Frames")
        frames_placeholder = st.empty()
        
        st.subheader("Recognition Result")
        result_text = st.empty()
        
        st.subheader("Recognition History")
        results_placeholder = st.empty()

    # Initialize camera and model
    initialize_camera()

    # Main loop
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Unable to read camera feed")
            break

        # Resize frame
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        
        # Display real-time feed (whether recording or not)
        camera_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # If recording, save frames
        if st.session_state.recording_active:
            print("Recording frames: ", len(st.session_state.recorded_frames))
            st.session_state.recorded_frames.append(frame)
            # Limit saved frames to avoid memory overflow
            if len(st.session_state.recorded_frames) > 30:  # Save at most 30 frames
                st.session_state.recorded_frames.pop(0)
        
        # Display processed frames and results
        if st.session_state.processed_frames is not None:
            combined_frame = np.hstack(st.session_state.processed_frames)
            frames_placeholder.image(combined_frame, channels="BGR", use_container_width=True)
            result_text.markdown(f"**Recognition Result:** {st.session_state.processing_result}")
        
        # Display recognition history
        if st.session_state.recognition_results:
            results_text = ["Recognition History:"]
            for i, result in enumerate(st.session_state.recognition_results):
                results_text.append(f"{i+1}. {result}")
            results_placeholder.markdown("\n".join(results_text))
        
        # Control frame rate
        time.sleep(0.33)

if __name__ == "__main__":
    main() 
    