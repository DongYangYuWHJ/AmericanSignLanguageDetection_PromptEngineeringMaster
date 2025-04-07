# Sign Language Recognition System

A try to utilize prompt engineering to mimic the effect of fine-tuning
only use gemini API to identify sign language that gemini don't know before

## Features

- Real-time camera feed for sign language gesture capture
- Recording functionality to capture sign language gestures
- AI-powered recognition of sign language gestures
- Support for some sign language words (CAT, ME, HELLO, DONE, LEARN, FINE)
- User-friendly interface with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. The application will open in your default web browser.

3. Use the control panel in the sidebar to:
   - Start recording: Click "Start Recording" to begin capturing sign language gestures
   - Stop recording and recognize: Click "Stop Recording and Recognize" to process the captured gestures
   - Clear results: Click "Clear Results" to reset the recognition history
   - Exit: Click "Exit" to close the application

4. The recognition results will be displayed in the right panel.

## Project Structure

- `main.py`: Main application file with Streamlit UI and camera handling
- `prompt_utils.py`: Utilities for interacting with the Gemini AI API
- `config.py`: Configuration settings for the application
- `.env`: Environment variables (API keys)
- `requirements.txt`: List of Python dependencies

## Requirements

- Python 3.8+
- OpenCV
- Streamlit
- Google Generative AI
- NumPy
- Pillow
- python-dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for providing the sign language recognition capabilities
- Streamlit for the web application framework
- OpenCV for computer vision capabilities
