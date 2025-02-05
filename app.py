import streamlit as st
import nltk
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import pyttsx3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

chatbot = pipeline("text-generation", model="distilgpt2")

recognizer = sr.Recognizer()

def get_tts_engine():
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = pyttsx3.init()
    return st.session_state.tts_engine

def analyze_sentiment(user_input):
    score = sia.polarity_scores(user_input)
    if score['compound'] < -0.3:
        return "It seems like you're feeling unwell or concerned. Please seek professional help if needed."
    return None

def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    
    if "symptom" in user_input:
        return "For general symptoms, try getting enough rest and staying hydrated. For severe cases, consult a doctor."
    
    elif "appointment" in user_input:
        return "Would you like to book an appointment?"
    
    elif "yes" in user_input:
        return "Please enter your preferred date and time for the appointment."
    
    elif "medication" in user_input:
        return "Make sure to take prescribed medicines on time. Would you like a medication reminder?"
    
    elif "health tips" in user_input:
        return "Tip: Stay hydrated, exercise regularly, and get at least 7-8 hours of sleep!"
    
    elif "emergency" in user_input:
        return "If this is a medical emergency, please call an ambulance or visit the nearest hospital immediately!"
    
    sentiment_response = analyze_sentiment(user_input)
    if sentiment_response:
        return sentiment_response

    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

def speech_input():
    with sr.Microphone() as source:
        st.session_state.listening = True
        st.write("Listening... Speak now.")
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)
            st.session_state.listening = False
            return user_input
        except sr.UnknownValueError:
            st.session_state.listening = False
            return "Sorry, I couldn't understand your speech."
        except sr.RequestError:
            st.session_state.listening = False
            return "Error with speech recognition service."

def speak_response(response):
    tts_engine = get_tts_engine()
    tts_engine.say(response)
    tts_engine.runAndWait()

def set_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stChatMessage {
            background-color: #2e2e2e;
            color: #ffffff; /* White text color */
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stChatMessage .stMarkdown {
            color: #ffffff !important; /* Ensure markdown text is white */
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            background-color: #2e2e2e;
            color: #ffffff;
            border-radius: 5px;
            padding: 10px;
        }
        .stSidebar {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot 🏥🤖", page_icon="🏥")
    set_dark_theme()  # Apply dark theme
    
    st.title("Healthcare Assistant Chatbot 🏥🤖")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.chat_input("How can I assist you today?")
    with col2:
        if st.button("🎤 Speak"):
            user_input = speech_input()

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Processing your query, please wait..."):
            response = healthcare_chatbot(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        speak_response(response)

    st.sidebar.title("💡 Health Tips")
    st.sidebar.markdown("""
        - ✅ Stay hydrated 💧
        - ✅ Exercise regularly 🏃‍♂️
        - ✅ Eat a balanced diet 🥗
        - ✅ Get enough sleep 💤
    """)

if __name__ == "__main__":
    main()