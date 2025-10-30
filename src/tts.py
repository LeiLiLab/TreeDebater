import json
import argparse
import os
import shutil
import re
import torch
from mutagen.mp3 import MP3
from io import BytesIO
from pathlib import Path
from openai import OpenAI

from utils.time_estimator import LengthEstimator
from utils.tool import remove_citation, remove_subtitles
from utils.constants import openai_api_key

import speech_recognition as sr
from pydub import AudioSegment

# python tts.py -i ../log_files/1.json -o ../results/audio -name case1_1

def get_options():
    parser = argparse.ArgumentParser(description='Text-to-Speech options')
    parser.add_argument('-i', type=str, help='input file path')
    parser.add_argument('-o', type=str, default="../results/audio",help='output directory path')
    parser.add_argument('-name', type=str, default="case1_1",help='output name')
    return parser.parse_args()

# I find different voices have very similar speech duration, so just use one voice for estimation
def query_time(content):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=content[:4096]
    )
    audio_bytes = BytesIO(response.content)
    
    return MP3(audio_bytes).info.length

    # speech_file_path = "temp.mp3"
    # response.stream_to_file(speech_file_path)
    # return MP3(speech_file_path).info.length

def convert_text_to_speech(content, output_path, voice="echo"):
    client = OpenAI()
    audio_content, reference = remove_citation(content)
    audio_content = remove_subtitles(audio_content)

    response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=audio_content[:4096],  # 4096 is the max limit
            # instructions=instructions,
            response_format="mp3",
        )

    response.stream_to_file(output_path)

    audio_bytes = BytesIO(response.content)
    duration = MP3(audio_bytes).info.length

    text_content, reference = remove_citation(content, keep_main=True)

    return text_content, reference, duration


def convert_debate_to_speech(input_file, output_dir, name):
    with open(input_file, "r") as file:
        data = json.load(file)
    
    config = data["config"]
    # title = config["env"]["motion"].replace(" ", "_").lower().replace("'", "").replace(",", "").replace(".", "")
    output_path = f"{output_dir}/{name}"
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(input_file, f"{output_path}/{name}.json")
    

    for stage in ["opening", "rebuttal", "closing"]:
    # for stage in ["closing"]:
        for side in ["for", "against"]:
        # for side in ["for"]:
            fname = f"{stage}_{side}"
            speech_file_path = f"{output_path}/{fname}.mp3"
            content = [x for x in data["debate_process"] if x["side"] == side and x["stage"] == stage]
            content = " ".join([x["content"] for x in content])
            print(f"Generating speech to {speech_file_path}")
            convert_text_to_speech(content, speech_file_path)
            print('word count & syllable count & speech length:', LengthEstimator(mode="words").query_time(content), LengthEstimator(mode="syllables").query_time(content), MP3(speech_file_path).info.length)
            print('word count per sec & syllable count per sec:', LengthEstimator(mode="words").query_time(content)/MP3(speech_file_path).info.length, LengthEstimator(mode="syllables").query_time(content)/MP3(speech_file_path).info.length)



from pydub import AudioSegment
import nltk
from pydub.silence import split_on_silence

def trim_audio_by_sentences(input_file, output_file, max_duration=240000):  # 240000 ms = 4 minutes
    """
    Trim an MP3 file to contain complete sentences within the max duration.
    
    Args:
        input_file (str): Path to input MP3 file
        max_duration (int): Maximum duration in milliseconds (default: 4 minutes)
    
    Returns:
        str: Path to the trimmed output file
    """
    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_file)
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()

    # Adjust recognition settings
    recognizer.energy_threshold = 300  # Increase sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Shorter pause threshold for better sentence detection
    
    # Split audio on silence to get rough chunks
    chunks = split_on_silence(
        audio,
        min_silence_len=500,  # minimum silence length (ms)
        silence_thresh=-40,    # silence threshold (dB)
        keep_silence=500       # keep some silence between chunks
    )
    
    # Process chunks and combine them within time limit
    trimmed_audio = AudioSegment.empty()
    trimmed_sentences = []
    
    for chunk in chunks:
        # Check if adding this chunk would exceed the time limit
        if len(trimmed_audio) + len(chunk) > max_duration:
            break
            
        # Convert chunk to wav for speech recognition
        chunk_wav = chunk.export(format="wav")
        
        try:
            # Perform speech recognition
            with sr.AudioFile(chunk_wav) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(
                    audio_data,
                    language="en-US",  # Specify language for better results
                    show_all=True      # Get detailed results
                )

                # Extract the most confident result
                if isinstance(text, dict) and 'alternative' in text:
                    text = text['alternative'][0]['transcript']
                else:
                    continue
                
                # Add basic punctuation based on pauses and speech patterns
                text = add_basic_punctuation(text)
                
                # Split text into sentences
                sentences = nltk.sent_tokenize(text, language="english")
                
                # If there's only one sentence in the chunk
                if len(sentences) == 1:
                    if len(trimmed_audio) + len(chunk) <= max_duration:
                        trimmed_audio += chunk
                        trimmed_sentences.append(sentences[0])
                else:
                    # Split chunk proportionally by sentence length
                    total_chars = len(text)
                    current_pos = 0
                    
                    for sentence in sentences:
                        sentence_ratio = len(sentence) / total_chars
                        sentence_duration = int(len(chunk) * sentence_ratio)
                        sentence_audio = chunk[current_pos:current_pos + sentence_duration]
                        
                        if len(trimmed_audio) + len(sentence_audio) <= max_duration:
                            trimmed_audio += sentence_audio
                            trimmed_sentences.append(sentence)
                        else:
                            # Stop if we can't add more sentences
                            break
                        
                        current_pos += sentence_duration
                
        except sr.UnknownValueError:
            # If speech recognition fails, treat chunk as a single unit
            if len(trimmed_audio) + len(chunk) <= max_duration:
                trimmed_audio += chunk
    
    # Export the trimmed audio
    trimmed_audio.export(output_file, format="mp3", bitrate="192k")
    
    # Print the duration of the trimmed audio
    duration_seconds = len(trimmed_audio) / 1000
    print(f"Trimmed audio duration: {duration_seconds:.2f} seconds")
    
    return duration_seconds, trimmed_sentences


def add_basic_punctuation(text):
    """
    Add basic punctuation based on common speech patterns and word cues.
    """
    # Common question words
    question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom'}
    
    # Split into word sequences
    words = text.split()
    result = []
    
    for i, word in enumerate(words):
        word = word.lower()
        next_word = words[i + 1].lower() if i + 1 < len(words) else ""
        
        # Add question marks
        if word in question_words and i == 0:
            words[-1] = words[-1] + "?"
            
        # Add periods for common sentence endings
        if i > 0 and i < len(words) - 1:
            prev_word = words[i - 1].lower()
            if prev_word in {'so', 'then', 'therefore', 'thus', 'hence', 'consequently'}:
                words[i - 2] = words[i - 2] + "."
                
        # Capitalize first word of apparent sentences
        if i == 0 or words[i - 1].endswith(('.', '?', '!')):
            word = word.capitalize()
            
        result.append(word)
    
    # Add final period if missing
    text = " ".join(result)
    if not text[-1] in {'.', '?', '!'}:
        text += "."
    
    return text

if __name__ == "__main__":
    input_file = "../log_files/1.json"
    output_dir = "../results/audio"
    name = "case1_1"
    convert_debate_to_speech(input_file, output_dir, name)