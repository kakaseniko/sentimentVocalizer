import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from pathlib import Path
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
import librosa
import wavio

def get_sentiment(text):

    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    text = text
    result = emotion_classifier(text)

    emotion_label = result[0]['label']
    return emotion_label

def _compute_embedding(audio):
  encoder.load_model("." / Path("encoder.pt"))
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, 22050))
  return embedding

def synthesize(embed, text):
  synthesizer = Synthesizer("." / Path("synthesizer.pt"))
  vocoder.load_model("." / Path("vocoder.pt"))
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
  return generated_wav

def get_voice(text, emotion_label):

   match emotion_label:
    case "amusment" | "excitement" | "joy" | "optimism" | "relief" | "love":
        audio_path = "./happy.wav"  #works but probably needs 22000 sample rate when saving
        sample_r = 21000
    case "anger" | "annoyance":
        audio_path = "./Carlin.mp3" #sample rate 18000
        sample_r = 18000
    case "sadness" | "grief" | "remorse" | "disappointment":
        audio_path = "./sad.wav" #works with 21000 sample rate
        sample_r = 21000
    case "nervousness":
        audio_path = "./Justin.mp3" #17000 sample rate
        sample_r = 17000
    case _:
        audio_path = "./neutral.wav" #works with 15500 sample rate
        sample_r = 15500

   audio_data, sample_rate = librosa.load(audio_path)
   embedding = _compute_embedding(audio_data)
   synth = synthesize(embedding, text)
   output_file = 'output.wav'
   wavio.write(output_file, synth, sampwidth=4, rate=sample_r)
   return "./output.wav"
 
st.write("""
# Sentiment vocalizer
""")

with st.form('my_form'):
  text = st.text_area('Enter text:', '')
  submitted = st.form_submit_button('Submit')
  
  if submitted:
    with st.spinner('Analyzing text...'):
       emotion_label = get_sentiment(text)
       st.info(emotion_label)

if submitted:
    with st.spinner('Generating audio...'):
        audio_file_path = get_voice(text, emotion_label)
        st.audio("./output.wav", format='audio/wav')

