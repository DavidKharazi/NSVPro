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
from .routes import router
from .models import init_metadata_db
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

app.mount("/", StaticFiles(directory="../chat-front/dist", html=True), name="static")


#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)


def check_new_files_in_s3(previous_files):
    """
    Проверяет наличие новых файлов в S3

    :param previous_files: Список ранее обнаруженных файлов
    :return: Список новых файлов и обновленный список всех файлов
    """
    current_files = []

    # Собираем файлы из разных источников (txt, json, docx, csv)
    for file_type in ['txt', 'json', 'docx', 'csv']:
        files = load_documents_local(BASE_DIRECTORY, FILE_TYPES)
        # files = load_s3_files(DATA_BUCKET, f'{current_user}/{file_type}/', f'.{file_type}')
        current_files.extend(files)

    # Находим новые файлы
    new_files = [file for file in current_files if file not in previous_files]

    return new_files, current_files


# def run_document_processing_cycle():
#     # Список ранее обработанных файлов
#     previous_files = []
#
#     while True:
#         try:
#             """Настройка расписания (например, загрузка каждые 6 часов (hours), минут (minutes) и т.д.)"""
#             schedule.every(1).minutes.do(download_csv)
#             print("Запущен планировщик для загрузки файла CSV.")
#             schedule.run_pending()
#             time.sleep(1)
#
#             # Проверяем наличие новых файлов
#             new_files, all_files = check_new_files_in_s3(previous_files)
#
#             # Если есть новые файлы
#             if new_files:
#                 print(f"Обнаружено новых файлов: {len(new_files)}")
#
#                 # 1. Загрузка документов (5 секунд максимум)
#                 start_time = time.time()
#                 DOCS = load_documents_local(BASE_DIRECTORY, FILE_TYPES)
#                 # DOCS = load_documents('s3', DATA_BUCKET, ['txt', 'json', 'docx', 'csv'])
#                 docs_time = time.time() - start_time
#                 print(f"Документы загружены за {docs_time:.2f} сек")
#
#                 if docs_time < 5:
#                     time.sleep(5 - docs_time)
#
#                 time.sleep(3)
#
#                 # 2. Нарезка документов на чанки (5 секунд максимум)
#                 start_time = time.time()
#                 chunks_res = split_docs_to_chunks(DOCS, ['txt', 'json', 'docx', 'csv'])
#                 chunks_time = time.time() - start_time
#                 print(f"Документы разделены на чанки за {chunks_time:.2f} сек")
#
#                 if chunks_time < 5:
#                     time.sleep(5 - chunks_time)
#
#                 time.sleep(3)
#
#                 current_chroma_path = (
#                     ChromaManager.CHROMA_PATH_SECONDARY
#                     if ChromaManager.USE_PRIMARY_CHROMA
#                     else ChromaManager.CHROMA_PATH_PRIMARY
#                 )
#                 print(f"CREATING VECTORSTORE IN: {current_chroma_path}")
#
#                 current_chroma_path = (
#                     ChromaManager.CHROMA_PATH_SECONDARY
#                     if ChromaManager.USE_PRIMARY_CHROMA
#                     else ChromaManager.CHROMA_PATH_PRIMARY
#                 )
#                 print(f"check_DB: {current_chroma_path}")
#
#                 # Создание новой vectorstore
#                 vectorstore_secondary = get_chroma_vectorstore(
#                     documents=chunks_res,
#                     embeddings=embeddings,
#                     persist_directory=current_chroma_path
#                 )
#
#                 # Устанавливаем флаг, что новая база готова
#                 ChromaManager.NEW_DATABASE_READY = True
#
#                 # Обновляем список обработанных файлов
#                 previous_files = all_files
#
#                 # Удаляем старую базу
#                 if current_chroma_path == f'./chroma/{current_user}/primary/':
#                     vectorstore = Chroma(persist_directory=ChromaManager.CHROMA_PATH_SECONDARY)
#                     vectorstore.delete_collection()
#                 else:
#                     vectorstore = Chroma(persist_directory=ChromaManager.CHROMA_PATH_PRIMARY)
#                     vectorstore.delete_collection()
#
#                 # Пауза для stabilization
#                 time.sleep(5)
#
#             else:
#                 # Если новых файлов нет, ждем 60 секунд перед следующей проверкой
#                 time.sleep(60)
#
#         except Exception as e:
#             print(f"Ошибка в цикле обработки документов: {e}")
#             # Небольшая пауза в случае ошибки
#             time.sleep(60)
#
#
# def run_server():
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)
#
#
#
# if __name__ == "__main__":
#     # Создаем два потока
#     document_processing_thread = threading.Thread(target=run_document_processing_cycle)
#     server_thread = threading.Thread(target=run_server)
#
#     # Запускаем потоки
#     document_processing_thread.start()
#     server_thread.start()
#
#     # Ждем завершения потоков
#     document_processing_thread.join()
#     server_thread.join()
#
#
def run_document_processing_cycle():
    # Список ранее обработанных файлов
    previous_files = []

    # Настройка расписания
    schedule.every(1).minutes.do(download_csv)
    print("Запущен планировщик для загрузки CSV файла.")

    while True:
        try:
            # Выполняем запланированные задачи
            schedule.run_pending()

            # Проверяем наличие новых файлов
            new_files, all_files = check_new_files_in_s3(previous_files)
            print(f"Новые файлы: {new_files}")

            # Если есть новые файлы
            if new_files:
                print(f"Обнаружено новых файлов: {len(new_files)}")

                # 1. Загрузка документов
                start_time = time.time()
                DOCS = load_documents_local(BASE_DIRECTORY, FILE_TYPES)
                print("Документы загружены локально.")
                docs_time = time.time() - start_time
                print(f"Документы загружены за {docs_time:.2f} сек")

                if docs_time < 5:
                    time.sleep(5 - docs_time)

                time.sleep(3)

                # 2. Нарезка документов на чанки
                start_time = time.time()
                chunks_res = split_docs_to_chunks(DOCS, ['txt', 'json', 'docx', 'csv'])
                print("Документы разделены на чанки.")
                chunks_time = time.time() - start_time
                print(f"Документы разделены на чанки за {chunks_time:.2f} сек")

                if chunks_time < 5:
                    time.sleep(5 - chunks_time)

                time.sleep(3)

                # Загрузка данных в Chroma
                try:
                    current_chroma_path = (
                        ChromaManager.CHROMA_PATH_SECONDARY
                        if ChromaManager.USE_PRIMARY_CHROMA
                        else ChromaManager.CHROMA_PATH_PRIMARY
                    )
                    print(f"Создаем Vectorstore в: {current_chroma_path}")

                    vectorstore_secondary = get_chroma_vectorstore(
                        documents=chunks_res,
                        embeddings=embeddings,
                        persist_directory=current_chroma_path
                    )

                    ChromaManager.NEW_DATABASE_READY = True
                    print(f"Данные успешно загружены в Chroma в {current_chroma_path}")
                except Exception as e:
                    print(f"Ошибка при загрузке данных в Chroma: {e}")

                # Обновляем список обработанных файлов
                previous_files = all_files
                time.sleep(5)

            else:
                time.sleep(60)

        except Exception as e:
            print(f"Ошибка в цикле обработки документов: {e}")
            time.sleep(60)

def run_server():
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

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