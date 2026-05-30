import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
import os

from symspellpy import SymSpell, Verbosity
from gtts import gTTS
from playsound import playsound
from deep_translator import GoogleTranslator

# ==========================================================
#                    LOAD MODEL
# ==========================================================

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ==========================================================
#                    SYMSPELL
# ==========================================================

sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7
)

sym_spell.load_dictionary(
    "frequency_dictionary_en_82_765.txt",
    0,
    1
)

# ==========================================================
#        ODIA TRANSLITERATION (FOR SCREEN DISPLAY)
# ==========================================================

odia_map = {

    "hello": "Namaskara",

    "good morning": "Suprabhat",

    "thank you": "Dhanyabad",

    "how are you": "Apana kemiti achanti",

    "goodbye": "Bidaya",

    "welcome": "Swagata",

    "i am fine": "Mu bhala achi",

    "please": "Dayakari",

    "sorry": "Khyama karantu",

    "yes": "Han",

    "no": "Na",

    "good night": "Subharatri"
}

# ==========================================================
#           REAL ODIA (FOR SPEECH OUTPUT)
# ==========================================================

odia_speech_map = {

    "hello": "ନମସ୍କାର",

    "good morning": "ସୁପ୍ରଭାତ",

    "thank you": "ଧନ୍ୟବାଦ",

    "how are you": "ଆପଣ କେମିତି ଅଛନ୍ତି",

    "goodbye": "ବିଦାୟ",

    "welcome": "ସ୍ୱାଗତ",

    "i am fine": "ମୁଁ ଭଲ ଅଛି",

    "please": "ଦୟାକରି",

    "sorry": "କ୍ଷମା କରନ୍ତୁ",

    "yes": "ହଁ",

    "no": "ନା",

    "good night": "ଶୁଭରାତ୍ରୀ"
}

# ==========================================================
#              SPELL CORRECTION FUNCTION
# ==========================================================

def correct_and_suggest(text):

    words = text.split()

    if not words:
        return text, []

    last_word = words[-1]

    suggestions = sym_spell.lookup(
        last_word,
        Verbosity.TOP,
        max_edit_distance=2
    )

    if suggestions:

        corrected_word = suggestions[0].term

        suggestions_list = [
            s.term for s in suggestions[:3]
        ]

    else:

        corrected_word = last_word
        suggestions_list = []

    words[-1] = corrected_word

    return " ".join(words), suggestions_list

# ==========================================================
#                    ODIA SPEECH
# ==========================================================

def speak(text):

    if text.strip() == "":
        return

    filename = "voice.mp3"

    try:

        # REAL ODIA FOR SPEECH
        translated = odia_speech_map.get(
            text.strip().lower(),
            None
        )

        # If not found in dictionary
        if translated is None:

            translated = GoogleTranslator(
                source='en',
                target='or'
            ).translate(text)

        print("Odia Speech:", translated)

        # Hindi voice reads Odia Unicode well
        tts = gTTS(
            text=translated,
            lang='hi'
        )

        tts.save(filename)

        playsound(filename)

        os.remove(filename)

    except Exception as e:

        print("Speech Error:", e)

# ==========================================================
#               NEXT WORD PREDICTION
# ==========================================================

common_phrases = {

    "hello": ["world", "there"],

    "how": ["are", "to"],

    "i": ["am", "have"],

    "you": ["are", "can"],

    "thank": ["you"],

    "good": ["morning", "job"]
}

def predict_next_word(text):

    words = text.lower().split()

    if not words:
        return []

    return common_phrases.get(words[-1], [])

# ==========================================================
#                    MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    model_complexity=0,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)

# ==========================================================
#                    CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================================
#                    TEXT SYSTEM
# ==========================================================

text = ""

corrected_text = ""

odia_display_text = ""

suggestions = []

next_words = []

prediction_history = []

stable_prediction = ""

stable_start_time = None

last_add_time = 0

last_correct_time = 0

# ==========================================================
#                    BUTTONS
# ==========================================================

btn_stop = (500, 10, 630, 60)

btn_clear = (350, 10, 480, 60)

btn_speak = (200, 10, 330, 60)

stop_clicked = False

# ==========================================================
#                  MOUSE CALLBACK
# ==========================================================

def mouse_callback(event, x, y, flags, param):

    global stop_clicked
    global text

    if event == cv2.EVENT_LBUTTONDOWN:

        # STOP
        if (
            btn_stop[0] <= x <= btn_stop[2]
            and
            btn_stop[1] <= y <= btn_stop[3]
        ):

            stop_clicked = True

        # CLEAR
        if (
            btn_clear[0] <= x <= btn_clear[2]
            and
            btn_clear[1] <= y <= btn_clear[3]
        ):

            text = ""

        # SPEAK
        if (
            btn_speak[0] <= x <= btn_speak[2]
            and
            btn_speak[1] <= y <= btn_speak[3]
        ):

            speak(corrected_text)

# ==========================================================
#                  WINDOW SETUP
# ==========================================================

cv2.namedWindow("Gesture AI Keyboard")

cv2.setMouseCallback(
    "Gesture AI Keyboard",
    mouse_callback
)

# ==========================================================
#                    MAIN LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(frame_rgb)

    current_prediction = None

    # ======================================================
    #                    BUTTONS
    # ======================================================

    cv2.rectangle(
        frame,
        btn_stop[:2],
        btn_stop[2:],
        (0,0,255),
        -1
    )

    cv2.putText(
        frame,
        "STOP",
        (510,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.rectangle(
        frame,
        btn_clear[:2],
        btn_clear[2:],
        (255,0,0),
        -1
    )

    cv2.putText(
        frame,
        "CLEAR",
        (360,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.rectangle(
        frame,
        btn_speak[:2],
        btn_speak[2:],
        (0,255,0),
        -1
    )

    cv2.putText(
        frame,
        "SPEAK",
        (210,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,0),
        2
    )

    # ======================================================
    #                  HAND DETECTION
    # ======================================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            x_ = []
            y_ = []

            data_aux = []

            for lm in hand_landmarks.landmark:

                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:

                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:

                prediction = model.predict(
                    [np.asarray(data_aux)]
                )[0]

                current_prediction = str(prediction)

                prediction_history.append(
                    current_prediction
                )

                if len(prediction_history) > 15:

                    prediction_history.pop(0)

                if len(prediction_history) >= 5:

                    stable_prediction = prediction_history[-5]

                cv2.putText(
                    frame,
                    f"Prediction: {stable_prediction}",
                    (10,170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )

                # ==================================================
                #              HOLD TIMER SYSTEM
                # ==================================================

                if stable_start_time is None:

                    stable_start_time = time.time()

                else:

                    elapsed = time.time() - stable_start_time

                    cv2.putText(
                        frame,
                        f"Hold: {elapsed:.1f}s",
                        (10,220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,255),
                        2
                    )

                    # Hold for 6 seconds
                    if elapsed > 6:

                        current_time = time.time()

                        if current_time - last_add_time > 2:

                            text += stable_prediction

                            last_add_time = current_time

                            prediction_history.clear()

                            stable_start_time = None

                            print("Added:", stable_prediction)

    else:

        stable_start_time = None

    # ======================================================
    #         SPELL CORRECTION + TRANSLITERATION
    # ======================================================

    if time.time() - last_correct_time > 3:

        corrected_text, suggestions = correct_and_suggest(text)

        if text.endswith(" "):

            text = corrected_text

        next_words = predict_next_word(corrected_text)

        # Transliteration display
        odia_display_text = odia_map.get(
            corrected_text.strip().lower(),
            corrected_text
        )

        last_correct_time = time.time()

    # ======================================================
    #                  DISPLAY TEXT
    # ======================================================

    cv2.putText(
        frame,
        f"Text: {text}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Corrected: {corrected_text}",
        (10,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,200,255),
        2
    )

    cv2.putText(
        frame,
        f"Odia: {odia_display_text}",
        (10,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,255),
        2
    )

    cv2.putText(
        frame,
        "Hold gesture for 6 sec to add",
        (10,430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    # ======================================================
    #              NEXT WORD PREDICTION
    # ======================================================

    for i, word in enumerate(next_words):

        cv2.putText(
            frame,
            f"> {word}",
            (250,260 + i*30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    # ======================================================
    #                    SHOW FRAME
    # ======================================================

    cv2.imshow(
        "Gesture AI Keyboard",
        frame
    )

    # ======================================================
    #                  KEYBOARD CONTROL
    # ======================================================

    key = cv2.waitKey(1) & 0xFF

    # Manual Space
    if key == ord('s'):

        text += " "

    # Clear
    if key == ord('c'):

        text = ""

    # Backspace
    if key == ord('b'):

        text = text[:-1]

    # Exit
    if (
        key == ord('q')
        or
        key == 27
        or
        stop_clicked
    ):
        break

# ==========================================================
#                    CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()