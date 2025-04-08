from collections import deque

SEQ_LEN = 30
SEQUENCE = deque(maxlen=SEQ_LEN)
LATEST_RESULT = None

def get_latest_result():
    global LATEST_RESULT
    result = 'Waiting for result ... ...' if LATEST_RESULT is None else LATEST_RESULT
    return {'result-text': result}

def send_video_to_api():
    # blocking, run in a separate thread
    global SEQUENCE, LATEST_RESULT

    def dummy_function(video):
        # Dummy function to simulate API call, add your logic here
        import random
        return random.choice(['Hello', 'World', 'Python', 'OpenCV'])
    
    while True:
        if len(SEQUENCE) == SEQ_LEN:
            video_data = list(SEQUENCE)
            LATEST_RESULT = dummy_function(video_data)

def process_sequence(frame):
    global SEQUENCE, SEQ_LEN
    seq, lim = SEQUENCE, SEQ_LEN
    if len(seq) < lim:
        seq.append(frame)
    else:
        # if maxlen is reached, pop the oldest frame and append the latest new one
        seq.popleft()
        seq.append(frame)
