import os
import sys
import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

### ローカル実行の場合の環境変数読み込み
# from dotenv import load_dotenv
# load_dotenv()

### Sreamlit Cloudにデプロイする場合の環境変数読み込み
os.environ['OPENAI_API_KEY'] = st.secrets.OpenAIAPI.openai_api_key
os.environ['GOOGLE_CSE_ID'] = st.secrets.GoogleCSE.google_cse_id
os.environ['GoogleAPI'] = st.secrets.google_api_key.google_api_key

def load_youtube(youtube_url):
    loader = YoutubeLoader.from_youtube_url(youtube_url, language="ja")
    transcript_text = loader.load()[0].page_content
    print(f"{transcript_text = }")
    print(f"{len(transcript_text) = }")

    text_spritter = CharacterTextSplitter(separator=" ", chunk_size=1000)
    texts = text_spritter.split_text(transcript_text)

    print(f"{len(texts) = }")

    docs = [Document(page_content=t) for t in texts]

    return docs

def run_summary(docs):
    chain = load_summarize_chain(
        llm=chat,
        chain_type="map_reduce",
        verbose=True,
    )

    output = chain(
        inputs=docs,
        return_only_outputs=True,
    )["output_text"]

    return jp_translation(output)

def run_qa(docs, question):

    chain = load_qa_chain(
        llm=chat,
        chain_type="map_reduce",
        verbose=True,
    )

    output = chain(
        {
            "input_documents": docs,
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]

    # return jp_translation(output)
    return output

def run_net_search(question, memory):
    # ChatGPT-3.5のモデルのインスタンスの作成
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = load_tools(["google-search"], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True)

    output = agent.run(question)

    output = jp_translation(output)

    return output, memory

def jp_translation(input):
    print('和訳前：',input)
    output = chat([HumanMessage(
        content=f"""
        次の文章を和訳して。
        {input}
        """)
    ]).content
    print('和訳後：',output)
    return output

# 画面UI定義-------------------------------------------------------------------------------------
# Streamlitによって、タイトル部分のUIをの作成
st.title("YouTube 動画要約 + Q&A")

# 入力フォームと送信ボタンのUIの作成
youtube_url = st.text_input("Youtbe 動画の URL を入力してください。")
summary_button = st.button("要約")
text_input = st.text_input("YouTube 動画の内容についての質問を入力してください。")

# ボタンを横並びに配置
col1, col2 = st.columns([1, 9]) 
with col1:
    qa_button = st.button("Q&A")
with col2:
    net_search_button = st.button("ネット検索")

# 要約内容を表示するサイドバーの作成
st.sidebar.title("要約内容")
# -------------------------------------------------------------------------------------------

# 要約内容を格納する配列の初期化
try:
    summary = st.session_state["summary"]
except:
    summary = ''

# チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
try:
    history = st.session_state["history"]
except:
    history = []

# メモリー情報を格納する配列の初期化
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

# ChatGPT-3.5のモデルのインスタンスの作成
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# YouTubeのURLが入力されている場合、字幕情報を読み込む
docs = ''
if youtube_url:
    try:
        docs = load_youtube(youtube_url)
    except:
        pass

# 要約ボタンが押された場合
if summary_button:
    summary_button = False

    if not youtube_url or not docs:
        # エラーメッセージを出力
        message('YouTube の URL を正しく入力してください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    else:
        # 要約を実行
        summary = run_summary(docs)
        st.session_state["summary"] = summary

# サイドバーに要約結果を表示
st.sidebar.write(summary)

# Q&Aボタンが押された場合
if qa_button:
    qa_button = False

    if not youtube_url or not docs:
        # エラーメッセージを出力
        message('YouTube の URL を正しく入力してください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    elif not text_input:
        # エラーメッセージを出力
        message('質問を入力してください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    else:
        # Q&Aを実行
        output = run_qa(docs, text_input)

        # 実行結果をチャット履歴に追加
        history.append({'msg_type':'HumanMessage', 'content':text_input})
        history.append({'msg_type':'AIMessage', 'content':output})
        # セッションへのチャット履歴の保存
        st.session_state["history"] = history

# ネット検索ボタンが押された場合
if net_search_button:
    net_search_button = False

    if not text_input:
        # エラーメッセージを出力
        message('質問を入力してください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    else:
        # ネット検索を実行
        output, memory = run_net_search(text_input, memory)

        # セッションへのメモリー情報の保存
        st.session_state["memory"] = memory

        # 実行結果をチャット履歴に追加
        history.append({'msg_type':'HumanMessage', 'content':text_input})
        history.append({'msg_type':'AIMessage', 'content':output})
        # セッションへのチャット履歴の保存
        st.session_state["history"] = history

# チャット履歴の表示
for index, chat_message in enumerate(reversed(history)):
    if chat_message['msg_type'] == 'HumanMessage':
        message(chat_message['content'], is_user=True, key=2 * index, avatar_style="micah", seed='Felix')
    elif chat_message['msg_type'] == 'AIMessage':
        message(chat_message['content'], is_user=False, key=2 * index + 1, avatar_style="bottts-neutral", seed='Aneka')
