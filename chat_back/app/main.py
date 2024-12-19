import os

import uvicorn
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from app.routes import router
from app.models import init_metadata_db
from services.chat import SQLiteChatHistory
from agents.agent import triage_agent
from services.database import ChromaManager
from services.vectorestore import (split_docs_to_chunks, load_documents_local,
                                   get_chroma_vectorstore, embeddings, current_user,
                                   BASE_DIRECTORY, FILE_TYPES)
import threading
import time
from agents.tools import download_csv
import schedule




model_name = "gpt-4o-mini"
temperature = 0
llm = ChatOpenAI(model=model_name, temperature=temperature)


# Укажите URL для загрузки файла
# CSV_URL = "https://nsv.by/dev/aicsv.php?key=cJT3qhgB9L9SjUmOko&ssid=1oAalTb506IyIonIFZLvgCa7LQ1VzsQQ&GEZhyH9Z=q0HR214dKNBFQHlHiIOZrlDxeCjyrmKWNbVhFqhZCPylzeRKHb"
CSV_URL = "https://nsv.by/dev/aicsv.php?key=cJT3qhgB9L9SjUmOko&ssid=1oAalTb506IyIonIFZLvgCa7LQ1VzsQQ&GEZhyH9Z=q0HR214dKNBFQHlHiIOZrlDxeCjyrmKWNbVhFqhZCPylzeRKHb"


# Укажите путь для папки, где будут храниться файлы
DATA_FOLDER = "data"
DATA_FOLDER_JSON = "upload_files"

# Укажите путь к вашему CSV файлу
csv_file_path = 'data/aiCSV.csv'

# Создание папок, если их нет
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER_JSON, exist_ok=True)


load_dotenv()
oauth = OAuth()
init_metadata_db()
# download_csv()



chat_history_for_chain = SQLiteChatHistory()



prompt_new = ChatPromptTemplate.from_messages(
    [
        (
            "system", triage_agent.instructions,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

chain_new = prompt_new | llm

chain_with_message_history = RunnableWithMessageHistory(
    chain_new,
    lambda session_id: chat_history_for_chain.messages(limit=0),
    input_messages_key="question",
    history_messages_key="chat_history",
)




app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/", StaticFiles(directory="../chat-front/dist", html=True), name="static")
app.mount("/", StaticFiles(directory="./dist", html=True), name="static")


def check_new_files(previous_files, current_files):
    # Извлекаем только имена файлов, без полных путей
    previous_file_names = {os.path.basename(f) for f in previous_files}
    current_file_names = {os.path.basename(f) for f in current_files}

    new_files = current_file_names - previous_file_names
    return [f for f in current_files if os.path.basename(f) in new_files]




def run_document_processing_cycle():
    """
    Непрерывный цикл обработки документов с мониторингом новых файлов
    """
    # Список ранее обработанных файлов
    previous_files = []

    schedule.every().day.at("6:00").do(download_csv)
    schedule.every().day.at("9:00").do(download_csv)
    schedule.every().day.at("12:00").do(download_csv)
    schedule.every().day.at("15:35").do(download_csv)
    schedule.every().day.at("18:00").do(download_csv)
    schedule.every().day.at("21:00").do(download_csv)
    # schedule.every(1).minutes.do(download_csv)
    print("Запущен планировщик для загрузки файла CSV.")

    while True:
        try:
            # Запуск планировщика CSV
            schedule.run_pending()
            # time.sleep(60)

            # Получаем текущий список файлов
            current_files = []

            # Собираем файлы из разных источников
            docs_dict = load_documents_local(BASE_DIRECTORY, FILE_TYPES)

            # Извлекаем пути к файлам из словаря документов
            for file_type in FILE_TYPES:
                if docs_dict[file_type]:
                    if file_type == 'json':
                        current_files.extend([meta['source'] for meta in docs_dict['json_metadata']])
                    else:
                        current_files.extend([doc.source for doc in docs_dict[file_type]])

            print(f"Найдено файлов: {len(current_files)}")
            print("Список файлов:", current_files)

            # Находим новые файлы
            new_files = [file for file in current_files if file not in previous_files]

            # Если есть новые файлы
            if new_files:
                print(f"Обнаружено новых файлов: {len(new_files)}")
                print("Новые файлы:", new_files)

                # 1. Загрузка документов
                start_time = time.time()
                DOCS = load_documents_local(BASE_DIRECTORY, FILE_TYPES)
                docs_time = time.time() - start_time
                print(f"Документы загружены за {docs_time:.2f} сек")

                # Выравнивание времени загрузки
                if docs_time < 5:
                    time.sleep(5 - docs_time)

                time.sleep(3)

                # 2. Нарезка документов на чанки
                start_time = time.time()
                chunks_res = split_docs_to_chunks(DOCS, FILE_TYPES)
                chunks_time = time.time() - start_time
                print(f"Документы разделены на чанки за {chunks_time:.2f} сек")

                # Выравнивание времени нарезки
                if chunks_time < 5:
                    time.sleep(5 - chunks_time)

                time.sleep(3)

                # Определение текущего пути для Chroma
                current_chroma_path = (
                    ChromaManager.CHROMA_PATH_SECONDARY
                    if ChromaManager.USE_PRIMARY_CHROMA
                    else ChromaManager.CHROMA_PATH_PRIMARY
                )
                print(f"CREATING VECTORSTORE IN: {current_chroma_path}")

                # Создание новой vectorstore
                vectorstore_secondary = get_chroma_vectorstore(
                    documents=chunks_res,
                    embeddings=embeddings,
                    persist_directory=current_chroma_path
                )

                # Установка путей к Chroma
                ChromaManager.CHROMA_PATH_PRIMARY = f'./chroma/{current_user}/primary/'
                ChromaManager.CHROMA_PATH_SECONDARY = f'./chroma/{current_user}/secondary/'

                # Устанавливаем флаг, что новая база готова
                ChromaManager.NEW_DATABASE_READY = True
                if ChromaManager.NEW_DATABASE_READY:
                    # Переключаем базу
                    ChromaManager.USE_PRIMARY_CHROMA = not ChromaManager.USE_PRIMARY_CHROMA
                    # Сбрасываем флаг
                    ChromaManager.NEW_DATABASE_READY = False

                    print(f"SWITCHED TO PRIMARY: {ChromaManager.USE_PRIMARY_CHROMA}")

                # Обновляем список обработанных файлов
                previous_files = current_files

                # Удаление старой базы
                try:
                    if current_chroma_path == f'./chroma/{current_user}/primary/':
                        vectorstore = Chroma(persist_directory=ChromaManager.CHROMA_PATH_SECONDARY)
                        vectorstore.delete_collection()
                    else:
                        vectorstore = Chroma(persist_directory=ChromaManager.CHROMA_PATH_PRIMARY)
                        vectorstore.delete_collection()
                except Exception as delete_error:
                    print(f"Ошибка при удалении старой базы: {delete_error}")

                # Пауза для стабилизации
                time.sleep(5)

            else:
                # Если новых файлов нет, ждем 60 секунд перед следующей проверкой
                print("Новых файлов не обнаружено. Ожидание...")
                time.sleep(1000)


        except Exception as e:
            print(f"Критическая ошибка в цикле обработки документов: {e}")
            # Большая пауза в случае серьезной ошибки
            time.sleep(10)




def run_server():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Создаем два потока
    document_processing_thread = threading.Thread(target=run_document_processing_cycle)
    server_thread = threading.Thread(target=run_server)

    # Запускаем потоки
    document_processing_thread.start()
    server_thread.start()

    # Ждем завершения потоков
    document_processing_thread.join()
    server_thread.join()


