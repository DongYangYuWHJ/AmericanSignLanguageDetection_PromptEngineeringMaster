import base64
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
import time

# Base prompt (as model configuration)
words = "CAT， ME, HELLO, DONE, LEARN, FINE"

BASE_PROMPT = """You are a professional sign language recognition expert. Please carefully analyze the hand gesture features in each image and return the corresponding English word.
Return only the word itself, without any explanation.
Possible words include: {words}"""

# Detailed instructions (as supplementary for each call)
DETAILED_PROMPT = """These images are continuous sign language actions, with each image representing a gesture.
Please follow these rules for recognition:
1. Analyze the main features of the gesture (such as hand position, shape, movement direction, etc.)
2. Consider the context and continuity of the gesture
3. NEVER return any other words except the ones in the possible words list
4. Ensure recognition results match the standard sign language vocabulary"""

EXAMPLE_PROMPT = """
The following are examples for you to follow, please analyze the gesture in the image and return the corresponding English word. Output only the word itself, without any explanation.
"""

START_PROMPT = """
Now it is your turn to recognize the sign language in the following images. Answer only the word itself, without any explanation.
"""

# Sign language examples description
SIGN_EXAMPLES = """
Here are detailed descriptions of sign language gestures:

CAT: form an open 'F' handshape with your dominant hand, where the index finger and thumb are separated. Position your hand near the side of your cheek. Pull your hand away from your face while bringing the index finger and thumb together, mimicking the action of pulling a cat's whiskers. The palm starts facing towards your face and ends facing sideways.

ME: extend your index finger (the rest of the fingers remain curled). Point to the center of your chest without any movement; simply point to yourself. The palm faces towards your body.

HELLO: use an open 'B' handshape (fingers extended and together, thumb extended). Touch the fingertips to the side of your forehead, above your eyebrow. Move your hand away from your forehead in a saluting motion. The palm faces outward. A friendly facial expression is appropriate.

DONE: use both hands in open '5' handshapes (fingers spread). Hold your hands in front of you at chest level. Twist both hands outward simultaneously so that the palms change from facing you to facing away. The palms start facing you and end facing away. An affirmative nod or a facial expression indicating completion can accompany the sign.

LEARN: position your non-dominant hand open with the palm facing up (acting as a book or source of information); the dominant hand starts with an open relaxed hand. Touch the fingertips of the dominant hand to the palm of the non-dominant hand, then bring the dominant hand up to the forehead, closing the fingers together as if grasping information and placing it into your head. The dominant hand starts with the palm facing down and ends with fingertips touching the forehead; the non-dominant hand has the palm facing up. A focused facial expression can be used to indicate the act of learning.

FINE: use an open '5' handshape (fingers extended and spread). Touch the thumb to the center of the chest. Tap the thumb against the chest once or twice. The palm faces sideways or slightly outward. A neutral or positive facial expression is appropriate.
"""

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gen_prompt(leng=None):
    """Generate base prompt text"""
    return [BASE_PROMPT, DETAILED_PROMPT, EXAMPLE_PROMPT, START_PROMPT]

def init_persistent_chat(model):
    """Initialize persistent_chat, send all prompts at once"""
    print("Starting persistent_chat initialization...")
    chat = model.start_chat(history=[])
    
    # Combine all prompts into one
    combined_prompt = f"{BASE_PROMPT}\n\n{DETAILED_PROMPT}\n\n{EXAMPLE_PROMPT}\n\n{SIGN_EXAMPLES}\n\n{START_PROMPT}"
    
    # Send all prompts at once
    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat.send_message(combined_prompt)
            print("Combined prompt sent")
            break
        except Exception as e:
            if "ResourceExhausted" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Increase wait time
                print(f"API quota limit, waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            raise e
    
    # Record base prompt length
    chat.base_history_length = len(chat.history)
    print(f"\nBase prompt history length: {chat.base_history_length}")
    
    print("\n=== persistent_chat initialization complete ===")
    print(f"Base prompt history length: {chat.base_history_length}")
    print(f"Current history length: {len(chat.history)}")
    print("=== Initialization complete ===")
    
    return chat

def call_gemini_api(content_parts, persistent_chat):
    """Call Gemini API for sign language recognition"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = persistent_chat.send_message(content_parts)
            time.sleep(1)  # Increase wait time
            persistent_chat.history = persistent_chat.history[:persistent_chat.base_history_length]
            return response.text.strip()
        except Exception as e:
            if "ResourceExhausted" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Increase wait time
                print(f"API quota limit, waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            print(f"API call error: {str(e)}")
            return "API ERROR"

def process_images_with_api(frames, persistent_chat):
    """Process images and return recognition results"""
    try:
        # Convert images to PIL Image objects
        pil_images = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            pil_images.append(frame)
        
        # Send images directly
        return call_gemini_api(pil_images, persistent_chat)
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return "IMAGE ERROR"