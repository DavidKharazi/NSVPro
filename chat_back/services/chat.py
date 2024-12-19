import sqlite3
from langchain.schema import AIMessage, HumanMessage
from .vectorestore import current_user
from agents.agent import run_full_turn
from services.database import DatabaseManager, db_manager
from langchain_community.chat_message_histories import ChatMessageHistory
# from .vectorestore import retriever
import logging
from fastapi import WebSocket, status




# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_by_email(email: str):
    with sqlite3.connect('metadata.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, cyberman_id FROM Users WHERE email = ?", (email,))
        return cursor.fetchone()




# Функция для поиска сессии по user_id и cyberman_id
def get_session_by_user_and_cyberman(user_id: int, cyberman_id: int):
    with sqlite3.connect('metadata.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM Session WHERE user_id = ? AND cyberman_id = ?", (user_id, cyberman_id))
        return cursor.fetchone()


class SQLiteChatHistory():
    def __init__(self, db_path="metadata.db", user_email=None, user_password=None, cyberman_name=None):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
        self.current_session_id = None
        self.user_email = user_email
        self.user_password = user_password
        self.cyberman_name = cyberman_name

    def start_new_session(self, user_email=None, user_password=None, cyberman_name=current_user, temperature=None, prompt_new=None):
        user_email = user_email or self.user_email
        user_password = user_password or self.user_password
        cyberman_name = cyberman_name or self.cyberman_name

        print(f"Starting new session with: email={user_email}, cyberman={cyberman_name}")


        # Получаем или создаем Cyberman и получаем его ID
        cyberman_id = self.db_manager.get_or_create_cyberman(cyberman_name, temperature, prompt_new)
        print(f"Cyberman ID: {cyberman_id}")

        # Передаем cyberman_id в метод get_or_create_user
        user_id = self.db_manager.get_or_create_user(user_email, user_password, cyberman_id)
        print(f"User ID: {user_id}")

        # Проверяем, существует ли уже сессия для данного пользователя и Cyberman
        session = self.db_manager.get_session_by_user_and_cyberman(user_id, cyberman_id)
        if session:
            self.current_session_id = session[0]
        else:
            self.current_session_id = self.db_manager.create_session(user_id, cyberman_id)
        print(f"Session ID: {self.current_session_id}")

        return self.current_session_id


    def add_message(self, message):
        if not self.current_session_id:
            print("No active session. Starting a new one.")
            self.start_new_session()

        print(f"Adding message to session {self.current_session_id}")

    def messages(self, limit=0):
        if not self.current_session_id:
            return ChatMessageHistory()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(f"SELECT * FROM Chat WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                  (self.current_session_id, limit))
        resp = c.fetchall()[::-1]
        chat_history = []
        for row in resp:
            id, message, sender, sent_at, session_id = row
            if sender == "human":
                chat_history.append(HumanMessage(content=message))
            elif sender == "ai":
                chat_history.append(AIMessage(content=message))
        conn.close()
        return ChatMessageHistory(messages=chat_history)

    def end_session(self):
        if self.current_session_id:
            self.db_manager.end_session(self.current_session_id)
            self.current_session_id = None


chat_history = SQLiteChatHistory()

chat_history_for_chain = SQLiteChatHistory()


async def process_message(state: dict, data: dict):
    """Обработка входящего сообщения"""
    websocket = state['websocket']
    retriever = state['retriever']  # Извлекаем retriever из state
    question_data = data.get('question_data')

    if not question_data:
        await websocket.send_json({"error": "Требуется question_data"})
        return

    question = question_data.get('question')
    if not question:
        await websocket.send_json({"error": "Требуется question"})
        return

    # Обновление состояния сессии если необходимо
    new_session_id = question_data.get('session_id')
    if new_session_id:
        state['session_id'] = new_session_id
        chat_history_for_chain.current_session_id = new_session_id



    # Добавляем новое сообщение в историю
    state['chat_history'].append({
        "role": "user",
        "content": question
    })

    state['chat_history'] = state['chat_history'][-3:]


    # Обработка сообщения через run_full_turn
    response = run_full_turn(
        state['current_agent'],
        state['chat_history'],  # Передаем полную историю чата
        retriever
    )

    print(f'response: {response}')

    # Обработка переключения агента
    if response.agent and response.agent != state['current_agent']:
        old_agent = state['current_agent']
        state['current_agent'] = response.agent
        switch_message = f"Переключение с {old_agent.name} на {response.agent.name}"
        logger.info(switch_message)

        # Очищаем историю при переключении агента
        # state['chat_history'] = []

        # Добавляем сообщение о переключении
        await websocket.send_json({
            "agent_switched": True,
            "current_agent": response.agent.name,
            "message": switch_message
        })

    # Получаем ответ от агента
    answer = response.messages[-1]['content'] if response.messages else None

    if answer:
        # Добавляем ответ агента в историю
        state['chat_history'].append({
            "role": "assistant",
            "content": answer
        })

        state['chat_history'] = state['chat_history'][-3:]

        # Сохранение сообщений
        chat_history_for_chain.add_message(HumanMessage(content=question))
        chat_history_for_chain.add_message(AIMessage(content=answer))

        db_manager.add_chat_message(state['session_id'], question, "human")
        db_manager.add_chat_message(state['session_id'], answer, "ai")

    # Отправка ответа
    await websocket.send_json({
        "answer": answer,
        "current_agent": state['current_agent'].name,
        "messages": db_manager.get_chat_messages_by_session_id(state['session_id'])
    })

    logger.info(f"Ответ отправлен пользователю. Текущий агент: {state['current_agent'].name}")


def validate_user_session(websocket: WebSocket, email: str) -> bool:
    """Проверяет валидность пользователя и его сессии"""
    if email is None:
        websocket.send_json({"error": "Требуется email"})
        websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    user = get_user_by_email(email)
    if user is None:
        websocket.send_json({"error": "Пользователь не найден"})
        websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    user_id, cyberman_id = user
    session = get_session_by_user_and_cyberman(user_id, cyberman_id)
    if session is None:
        websocket.send_json({"error": "Сессия не найдена"})
        websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    return True