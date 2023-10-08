import pandas as pd
import numpy as np
import whisper
import pytube
from pytube import YouTube
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import os
import pinecone
from dotenv import load_dotenv
import argparse
import pickle
import subprocess
import sys

# Create the argument parser
parser = argparse.ArgumentParser(description='')

# Add arguments
parser.add_argument('-l', '--link', type=str, help='URL to youtube video')
parser.add_argument('-k', '--key', type=str, help='Open AI Key')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
youtube_link = args.link
#youtube_link = 'https://www.youtube.com/watch?v=Hz7DW2cC3pc'

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

# Read key
try:
    user_secret = args.key
    if user_secret is None:
        sys.stop()
except Exception:
    user_secret = ''
    with open('openai.key', 'r') as file: 
        user_secret = file.readline().rstrip('\n')


youtube_video = YouTube(youtube_link)
video_title = youtube_video.title
video_date = youtube_video.publish_date.strftime("%Y-%m-%d")  # Format the publish date as YYYY-MM-DD

# Create directory with video title and date if it doesn't exist
directory = f"{video_date} - {video_title}"
if not os.path.exists(directory):
    os.makedirs(directory)

print(f"Using video called: {video_title}")


outpath = f'{directory}/youtube_video.mp4'
if not os.path.exists(outpath):
    try:    
        mp4_video = youtube_video.streams.filter(file_extension="mp4").get_by_resolution("720p").download(filename=outpath)

    except Exception:
        subprocess.run(["youtube-dl", "--recode-video", "mp4", youtube_link], check=True)
        mp4_files = [file for file in os.listdir(os.getcwd()) if file.endswith('.mp4')]

        # Construct the destination file path
        destination_file = os.path.join(directory, 'youtube_video.mp4')

        # Move the file to the destination directory with the new name
        os.rename(mp4_files[0], destination_file)
    # except Exception:
    #     video_id = pytube.extract.video_id(youtube_link)
    #     streams = youtube_video.streams.filter()
    #     stream = streams.first()

    #     mp4_video = stream.download(filename=f'{directory}/youtube_video.mp4')
    
    audio_file = open(mp4_video, 'rb')

print(f"Video downloaded in {outpath}")

file_path = f'{directory}/transcription.pkl'
if not os.path.exists(file_path):
    # Whisper
    output = model.transcribe(f'{directory}/youtube_video.mp4')
    with open(file_path, 'wb') as file:
        pickle.dump(output, file)
else:
    with open(file_path, 'rb') as file:
        output = pickle.load(file)

print("Geting transcription")

if not os.path.exists(f'{directory}/word_embeddings.csv'):
    # Transcription
    transcription = {
        "title": youtube_video.title.strip(),
        "transcription": output['text']
    }
    data_transcription.append(transcription)
    pd.DataFrame(data_transcription).to_csv(f'{directory}/transcription.csv')
    
segments = output['segments']

print("Geting embeddings")

if not os.path.exists(f'{directory}/word_embeddings.csv'):
    #Embeddings
    for segment in segments:
        openai.api_key = user_secret
        response = openai.Embedding.create(
            input= segment["text"].strip(),
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        meta = {
            "text": segment["text"].strip(),
            "start": segment['start'],
            "end": segment['end'],
            "embedding": embeddings
        }
        data.append(meta)

    # upsert_response = index.upsert(
    #         vectors=data,
    #         namespace=video_id
    #     )
    pd.DataFrame(data).to_csv(f'{directory}/word_embeddings.csv')

print("Done!")








