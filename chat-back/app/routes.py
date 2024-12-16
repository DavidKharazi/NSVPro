import re
from fastapi import Request, HTTPException, WebSocket, WebSocketDisconnect, Form, APIRouter
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from services.database import db_manager, add_user_to_db, ChromaManager
from services.chat import SQLiteChatHistory, validate_user_session, process_message
import sqlite3
from agents.agent import triage_agent
import logging
from services.chat import get_user_by_email, get_session_by_user_and_cyberman
from services.vectorestore import current_user, embeddings

router = APIRouter()

chat_history_for_chain = SQLiteChatHistory()

@router.get("/register", response_class=FileResponse)
async def get_register():
    return FileResponse("static/register.html")


def is_email_unique(email: str) -> bool:
    """Проверяет, является ли email уникальным в таблице Users."""
    with sqlite3.connect('metadata.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM Users WHERE email = ?", (email,))
        return cursor.fetchone() is None

@router.get("/confirm-email")
async def confirm_email():
    try:
        with sqlite3.connect('metadata.db') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, email FROM Users ORDER BY id DESC LIMIT 1")
            user = cursor.fetchone()

            if user:
                user_id, username = user

                # Обновление статуса пользователя
                cursor.execute(
                    "UPDATE Users SET is_active = TRUE WHERE id = ?",
                    (user_id,)
                )
                conn.commit()

                # Логика создания сессии и добавления сообщения
                chat_history = SQLiteChatHistory(user_email=username, user_password="dummy_password")
                print(f"SQLiteChatHistory создан с user_email={username}")
                try:
                    session_id = chat_history.start_new_session()
                    if not check_session_id_exists(session_id):
                        print(f"Session started: {session_id}")
                        db_manager.add_chat_message(session_id,
                                                    "Вас приветствует На Связи! Напишите Ваш вопрос.",
                                                    "Система")
                    else:
                        print(f"Session started: {session_id}")
                    # Переадресация на клиентскую часть после активации
                    return RedirectResponse(url="/")
                except Exception as e:
                    print(f"Error starting session: {e}")
                    return JSONResponse(content={"status": "error", "message": "Failed to start session"},
                                        status_code=500)
            else:
                # return JSONResponse(content={"status": "error", "message": "Invalid or expired token."}, status_code=400)
                return FileResponse("static/error_page.html")

    except Exception as e:
        print(f"Error confirming email: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


def validate_phone_number(phone: str) -> bool:
    # Проверка на формат: 9 цифр
    return bool(re.fullmatch(r"\d{9}", phone))


@router.post("/register", include_in_schema=False)  # Без слэша
@router.post("/register/")
async def post_register(user: str = Form(...), phone: str = Form(...)):
    # Ожидаем только 9 цифр
    if not validate_phone_number(phone):
        raise HTTPException(status_code=400, detail="Номер телефона должен содержать 9 цифр.")

    # Логика проверки уникальности номера телефона
    # if not is_email_unique(phone):  # username теперь без префикса
    #     return JSONResponse(content={"status": "error", "message": "Пользователь с таким телефоном уже существует."},
    #                         status_code=401)

    # Сохранение пользователя в базе данных
    user_id = add_user_to_db(phone, user)  # username теперь без префикса
    await confirm_email()

    return JSONResponse(
        content={"status": "success",
                 "message": "Здравствуйте! Меня зовут Олег, я - менеджер по товарам. Чем могу вам помочь?"},
        status_code=200
    )



# Функция для поиска по session_id в Chat
def check_session_id_exists(session_id: str) -> bool:
    with sqlite3.connect('metadata.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS(SELECT 1 FROM Chat WHERE session_id = ?)", (session_id,))
        return cursor.fetchone()[0] == 1



@router.post("/create_new_chat/")
async def create_new_chat(request: Request):

    data = await request.json()
    email = data.get('email')

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    user = get_user_by_email(email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user_id, cyberman_id = user
    session_id = db_manager.create_new_chat_session(user_id, cyberman_id)

    # Создаем начальное сообщение в чате с использованием нового session_id
    db_manager.add_chat_message(session_id, "Вас приветствует На Связи! Напишите Ваш вопрос.", "Система")

    return {"session_id": session_id}



@router.get("/get_chat_messages/{session_id}")
async def get_chat_messages(session_id: int):
    messages = db_manager.get_chat_messages_by_session_id(session_id)
    return {"messages": messages}


@router.get("/get_user_chats/{email}")
async def get_user_chats(email: str):
    user = get_user_by_email(email)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user_id, _ = user
    chats = db_manager.get_chats_by_user_id(user_id)
    return {"chats": chats}


def get_session_history(session_id):
    history = chat_history_for_chain.messages(limit=0)
    print(f"Session {session_id} history: {history}")
    return history




# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#
# @router.websocket("/ws/rag_chat/")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#
#     # Валидация пользователя
#     email = websocket.query_params.get('email')
#     if not validate_user_session(websocket, email):
#         return
#
#     user_id, cyberman_id = get_user_by_email(email)
#     session = get_session_by_user_and_cyberman(user_id, cyberman_id)
#     session_id = session[0]
#     chat_history_for_chain.current_session_id = session_id
#
#     # Инициализация состояния
#     state = {
#         'current_agent': triage_agent,
#         'session_id': session_id,
#         'websocket': websocket,
#         'chat_history': []  # Добавляем историю чата в состояние
#     }
#
#     # Отправка исторических сообщений
#     messages = db_manager.get_chat_messages_by_session_id(session_id)
#     await websocket.send_json({"messages": messages})
#
#     try:
#         while True:
#             data = await websocket.receive_json()
#             await process_message(state, data)
#     except WebSocketDisconnect:
#         logger.info(f"Соединение с пользователем {email} закрыто. Завершаем сессию.")
#         chat_history_for_chain.end_session()


@router.websocket("/ws/rag_chat/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Валидация пользователя
    email = websocket.query_params.get('email')
    if not validate_user_session(websocket, email):
        return

    user_id, cyberman_id = get_user_by_email(email)
    session = get_session_by_user_and_cyberman(user_id, cyberman_id)
    session_id = session[0]
    chat_history_for_chain.current_session_id = session_id

    # Установка путей к Chroma
    ChromaManager.CHROMA_PATH_PRIMARY = f'./chroma/{current_user}/primary/'
    ChromaManager.CHROMA_PATH_SECONDARY = f'./chroma/{current_user}/secondary/'

    # Инициализация состояния
    state = {
        'current_agent': triage_agent,
        'session_id': session_id,
        'websocket': websocket,
        'chat_history': []  # Добавляем историю чата в состояние
    }

    try:
        while True:
            data = await websocket.receive_json()

            # Проверяем, готова ли новая база
            if ChromaManager.NEW_DATABASE_READY:
                # Переключаем базу
                ChromaManager.USE_PRIMARY_CHROMA = not ChromaManager.USE_PRIMARY_CHROMA
                # Сбрасываем флаг
                ChromaManager.NEW_DATABASE_READY = False

                print(f"SWITCHED TO PRIMARY: {ChromaManager.USE_PRIMARY_CHROMA}")

            # Получаем актуальный retriever при каждом запросе
            retriever = ChromaManager.get_current_retriever(embeddings)

            if retriever is None:
                await websocket.send_json({"error": "Не удалось загрузить базу данных"})
                continue

            # Добавляем retriever в state
            state['retriever'] = retriever

            await process_message(state, data)
    except WebSocketDisconnect:
        logger.info(f"Соединение с пользователем {email} закрыто. Завершаем сессию.")
        chat_history_for_chain.end_session()