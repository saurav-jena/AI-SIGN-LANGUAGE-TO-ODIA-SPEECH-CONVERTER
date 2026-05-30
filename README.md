# AI-SIGN-LANGUAGE-TO-ODIA-SPEECH-CONVERTER
AI Sign Language to Speech Converter is an intelligent assistive communication system designed to bridge the communication gap between hearing-impaired individuals and the general public.

The system uses MediaPipe Hands to extract 21 hand landmarks from live webcam input and employs a Random Forest classifier for real-time Indian Sign Language (ISL) gesture recognition. Recognized gestures are converted into text, refined using SymSpell-based spell correction, enhanced with next-word prediction, transliterated into Odia, and finally converted into speech using Google Text-to-Speech (gTTS).

Key Features:
• Real-time ISL gesture recognition
• MediaPipe-based hand landmark extraction
• Random Forest gesture classification
• Stable gesture holding mechanism
• Prediction history filtering
• SymSpell spell correction
• Next-word prediction
• Interactive smart keyboard interface
• Odia transliteration support
• English-to-Odia translation
• Real-time speech synthesis using gTTS
• Lightweight CPU-based implementation

Technologies Used:
Python, OpenCV, MediaPipe, Scikit-learn, NumPy, SymSpellPy, Deep Translator, gTTS, Playsound

This project provides an affordable, lightweight, and scalable communication solution for hearing and speech-impaired individuals while demonstrating the practical application of Artificial Intelligence, Computer Vision, Machine Learning, and Natural Language Processing.
