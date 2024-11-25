import os

import openai
from gtts import gTTS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# PDFファイルをロード
loader = PyPDFLoader("netevolve.pdf")
documents = loader.load()
system_message = SystemMessage(
    content=(
        "あなたは親切でプロフェッショナルなアシスタントです。"
        "ユーザーの言語に基づいて、回答を日本語または英語で行ってください。"
        "ユーザーが指定した言語で必ず回答してください。"
    )
)
# 埋め込み生成と検索のためのベクトルストアを作成
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# 質疑応答のためのLangChainのセットアップ
qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
    ),
    vector_store.as_retriever(),
)


# ユーザからの入力を受け取り、回答を生成するメインのループ
def main():
    print("\n--- 論文質疑応答システム ---")
    print("終了するには 'exit' と入力してください\n")
    conversation_history = []

    while True:
        user_question = input("あなた: ")
        if user_question.lower() == "exit":
            break

        # 回答の言語と形式を指示する追加プロンプト
        formatted_question = (
            "あなたはこの分野のプロフェッショナルです。以下の質問に対して、必要な項目を箇条書きで完結に答えてください。なるべく1回の回答で相手を説得させるように工夫してください。また、英語で回答してください。\n"
            f"質問: {user_question}"
        )

        # LangChainを使用して応答を生成
        result = qa_chain(
            {"question": formatted_question, "chat_history": conversation_history}
        )
        answer = result["answer"]
        print(f"AI: {answer}\n")
        # gTTSを使って回答を音声に変換
        tts = gTTS(text=answer, lang="en")
        tts.save("answer_en.mp3")

        # 音声を再生
        # os.system("start answer.mp3")  # Windowsの場合
        # os.system("afplay answer.mp3")  # macOSの場合
        # os.system("xdg-open answer.mp3")  # Linuxの場合
        # 会話履歴に追加
        conversation_history.append((user_question, answer))


if __name__ == "__main__":
    main()
