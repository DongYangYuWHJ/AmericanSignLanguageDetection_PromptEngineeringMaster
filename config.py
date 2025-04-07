import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Gesture recognition configuration
GESTURE_PROMPT = """
You are an expert in American Sign Language (ASL).

You will be shown one or more images of a single ASL hand gesture. These images are captured from a webcam and may contain slight variations in position or lighting.

Your task is to identify which ASL word is being shown in the image(s). Choose only from the following common ASL words:

[HELLO, MY, NAME, IS, YOU, WHAT, THANK, LOVE, YES, NO, PLEASE, HELP, GOOD, BAD, FRIEND, SORRY, WHERE, WHO, WHEN, HOW]

Please respond with only the identified ASL word in uppercase (e.g., "LOVE").
don't respond with any other words.
"""

# Sentence generation prompt
SENTENCE_PROMPT = """
Generate a complete English sentence based on the following sign language word sequence:
{words}

Please ensure the generated sentence follows English grammar rules and expresses complete meaning.
Return only the generated sentence, without any explanation.
"""

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_RATE = 30

# Gesture recognition threshold
GESTURE_THRESHOLD = 0.8  # Confidence threshold for gesture recognition 