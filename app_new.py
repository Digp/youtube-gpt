import os
import pandas as pd
import numpy as np
import streamlit as st
import whisper
import pytube
from pytube import YouTube
from streamlit_chat import message
import openai
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import pinecone
from dotenv import load_dotenv
import time
import subprocess

# whisper
model = whisper.load_model('base')
output = ''
data = []
data_transcription = []
embeddings = []
mp4_video = ''
audio_file = ''

# Pinacone

# Uncomment this section if you want to save the embedding in pinecone
#load_dotenv()
# initialize connection to pinecone (get API key at app.pinecone.io)
# pinecone.init(
#     api_key=os.getenv("PINACONE_API_KEY"),
#     environment=os.getenv("PINACONE_ENVIRONMENT")
# )
array = []

# Initialize session state variables if they don't exist
if 'youtube_link' not in st.session_state:
    st.session_state['youtube_link'] = ''

if 'user_secret' not in st.session_state:
    st.session_state['user_secret'] = ''

if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ''

# Get YouTube link and OpenAI API Key
youtube_link = st.text_input("Enter YouTube Link:", value=st.session_state['youtube_link'], key="youtube_link_input")
st.session_state['youtube_link'] = youtube_link

user_secret = st.text_input(label=":blue[OpenAI API key]",
                            placeholder="Paste your openAI API key, sk-",
                            type="password",
                            value=st.session_state['user_secret'],
                            key="api_key_input")
st.session_state['user_secret'] = user_secret

button_clicked = st.button("Start Analysis")
if button_clicked:
    st.session_state['button_clicked'] = True

# Check if both YouTube link and API key are provided
if st.session_state['youtube_link'] and st.session_state['user_secret'] and st.session_state['button_clicked']:
    st.markdown('<h1>Youtube GPT 🤖<small> by <a href="https://codegpt.co">Code GPT</a></small></h1>', unsafe_allow_html=True)
    #st.write("...")

    try:
        # Replicate code in script to get directory name
        youtube_video = YouTube(youtube_link)
        video_title = youtube_video.title
        video_date = youtube_video.publish_date.strftime("%Y-%m-%d")  # Format the publish date as YYYY-MM-DD
        directory = f"{video_date} - {video_title}"

        st.write(youtube_link)

        DEFAULT_WIDTH = 80
        #VIDEO_DATA = "https://youtu.be/bsFXgfbj8Bc"

        width = 40

        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, 47, side])
        #container.video(data=VIDEO_DATA)
        container.video(data=youtube_link)
    except Exception:
        st.write("...")


    tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
    with tab1:
        # Execute the code when the button is clicked
        st.write("Button is clicked! Running some code...")
        # Your code here
        with st.spinner('Running process...'):
            #st.write("# Get the video mp4")
            string = f'python script.py --link {youtube_link}'
            print(string)
            subprocess.call(string, shell=True)
            time.sleep(2)

        st.write("## Ready to chat!")
        #st.experimental_rerun() # Rerun the app to reload the tabs with the new directory info
        #st.write(youtube_link)

    with tab2: 
        st.header("Transcription:")
        if(os.path.exists(f'{directory}/youtube_video.mp4')):
            audio_file = open(f'{directory}/youtube_video.mp4', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
        if os.path.exists(f'{directory}/transcription.csv'):
            df = pd.read_csv(f'{directory}/transcription.csv')
            st.write(df)
    with tab3:
        st.header("Embedding:")
        print(f'{directory}/word_embeddings.csv')
        if os.path.exists(f'{directory}/word_embeddings.csv'):
            df = pd.read_csv(f'{directory}/word_embeddings.csv')
            st.write(df)
    with tab4:
        # user_secret = st.text_input(label = ":blue[OpenAI API key]",
        #                             placeholder = "Paste your openAI API key, sk-",
        #                             type = "password")
        #user_secret = st.session_state.get('user_secret', '')
        #st.write('To obtain an API Key you must create an OpenAI account at the following link: https://openai.com/api/')
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

        def get_text():
            if user_secret:
                st.header("Ask me something about the video:")
                input_text = st.text_input("You: ","", key="input")
                return input_text
        user_input = get_text()

        def get_embedding_text(api_key, prompt):
            openai.api_key = user_secret
            response = openai.Embedding.create(
                input= prompt.strip(),
                model="text-embedding-ada-002"
            )
            q_embedding = response['data'][0]['embedding']
            df=pd.read_csv(f'{directory}/word_embeddings.csv', index_col=0)
            df['embedding'] = df['embedding'].apply(eval).apply(np.array)

            df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
            returns = []
            
            # Sort by distance with 2 hints
            for i, row in df.sort_values('distances', ascending=True).head(4).iterrows():
                # Else add it to the text that is being returned
                returns.append(row["text"])

            # Return the context
            return "\n\n###\n\n".join(returns)

        # def generate_response(api_key, prompt):
        #     one_shot_prompt = '''I am YoutubeGPT, a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.
        #     Q: What is human life expectancy in the United States?
        #     A: Human life expectancy in the United States is 78 years.
        #     Q: '''+prompt+'''
        #     A: '''
        #     completions = openai.Completion.create(
        #         engine = "text-davinci-003",
        #         prompt = one_shot_prompt,
        #         max_tokens = 1024,
        #         n = 1,
        #         stop=["Q:"],
        #         temperature=0.2,
        #     )
        #     message = completions.choices[0].text
        #     return message
        
        def generate_response(api_key, prompt):
            one_shot_prompt = st.session_state['conversation_history'] + "You: " + prompt

            completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt=one_shot_prompt,
                max_tokens=1024,
                n=1,
                stop=["Q:"],
                temperature=0.2
            )
            message = completions.choices[0].text
            return message

        if user_input:
            text_embedding = get_embedding_text(user_secret, user_input)
            title = pd.read_csv(f'{directory}/transcription.csv')['title']
            string_title = "\n\n###\n\n".join(title)
            #user_input_embedding = 'Using this context: "'+string_title+'. '+text_embedding+'", answer the following question. \n'+user_input
            user_input_embedding = 'Using this context: "' + string_title + '. ' + text_embedding + '", answer the following question. \n' + youtube_link + ' ' + user_input
            # st.write(user_input_embedding)
            # Append user's input to the conversation history

            st.session_state['conversation_history'] += f'You: {user_input}\n'
            output = generate_response(user_secret, user_input_embedding)
            st.session_state['conversation_history'] += f'Bot: {output}\n'

            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

else:
    st.write("Please provide both a YouTube link and your OpenAI API key to continue.")
