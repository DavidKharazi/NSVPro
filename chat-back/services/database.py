import sqlite3
import uuid
from datetime import datetime
from fastapi import HTTPException
from langchain_community.vectorstores import Chroma
from .vectorestore import current_user


class DatabaseManager:
    def __init__(self, db_path="metadata.db"):
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по именам
        self.cursor = self.connection.cursor()
        self.db_path = db_path
        self.connection.row_factory = sqlite3.Row  # Позволяет работать с результатами в виде объектов Row

    def get_chats_by_user_id(self, user_id):
        query = """
        SELECT id, cyberman_id, started_at FROM Session WHERE user_id = ?
        """
        cursor = self.connection.cursor()
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def create_new_chat_session(self, user_id, cyberman_id=1):  # По умолчанию cyberman_id = 1
        query = """
            INSERT INTO Session (user_id, cyberman_id) VALUES (?, ?)
            """
        cursor = self.connection.cursor()
        cursor.execute(query, (user_id, cyberman_id))
        self.connection.commit()
        return cursor.lastrowid

    def get_chat_messages_by_session_id(self, session_id):
        query = """
            SELECT sender, messages, sent_at FROM Chat WHERE session_id = ?
            """
        cursor = self.connection.cursor()
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_or_create_user(self, email, name, cyberman_id=1):  # По умолчанию cyberman_id = 1
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM Users WHERE email = ?", (email,))
            user = cursor.fetchone()
            if user:
                print(f"Existing user found: {user[0]}")
                cursor.execute("UPDATE Users SET cyberman_id = ? WHERE email = ?", (cyberman_id, email))
                conn.commit()
                return user[0]
            else:
                confirmation_token = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO Users (email, name, cyberman_id, is_active, confirmation_token) VALUES (?, ?, ?, ?, ?)",
                    (email, name, cyberman_id, False, confirmation_token)
                )
                user_id = cursor.lastrowid
                print(f"New user created: {user_id}")
                return user_id, confirmation_token

    def get_or_create_cyberman(self, name, creativity, prompt=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM Cyberman WHERE name = ?", (name,))
            cyberman = cursor.fetchone()
            if cyberman:
                print(f"Existing cyberman found: {cyberman[0]}")
                return cyberman[0]
            else:
                cursor.execute("INSERT INTO Cyberman (name, creativity, prompt) VALUES (?, ?, ?)",
                               (current_user, creativity, prompt))
                cyberman_id = cursor.lastrowid
                print(f"New cyberman created: {cyberman_id}")
                return cyberman_id

    def create_session(self, user_id, cyberman_id=1):  # По умолчанию cyberman_id = 1
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            started_at = datetime.now()
            cursor.execute("INSERT INTO Session (started_at, user_id, cyberman_id) VALUES (?, ?, ?)",
                           (started_at, user_id, cyberman_id))
            session_id = cursor.lastrowid
            print(f"New session created: {session_id}")
            return session_id

    def end_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            ended_at = datetime.now()
            cursor.execute("UPDATE Session SET ended_at = ? WHERE id = ?", (ended_at, session_id))

    def add_chat_message(self, session_id, message, sender):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Chat (messages, sender, session_id) VALUES (?, ?, ?)",
                           (message, sender, session_id))

    def get_session_by_user_and_cyberman(self, user_id, cyberman_id=1):  # По умолчанию cyberman_id = 1
        query = """
        SELECT id FROM Session WHERE user_id = ? AND cyberman_id = ?
        """
        self.cursor.execute(query, (user_id, cyberman_id))
        return self.cursor.fetchone()

    def delete_chat(self, session_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Удаляем сообщения чата
            cursor.execute("DELETE FROM Chat WHERE session_id = ?", (session_id,))
            # Удаляем саму сессию
            cursor.execute("DELETE FROM Session WHERE id = ?", (session_id,))
            conn.commit()


db_manager = DatabaseManager()


def add_user_to_db(email: str, name: str, cyberman_id: int = 1, chat_id: int = None):

    try:
        with sqlite3.connect('metadata.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Users (email, name, cyberman_id, chat_id, is_active) VALUES (?, ?, ?, ?, ?)",
                (email, name, cyberman_id, chat_id, False)
            )
            conn.commit()
            user_id = cursor.lastrowid  # Получаем ID добавленного пользователя
            return user_id

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="User already registered")



def delete_chat_history_last_n(self, n=10):
    conn = sqlite3.connect(self.db_path)
    c = conn.cursor()
    c.execute(f'''
    with max_id as (select max(id) as maxid from history_messages where user_id = '{current_user}')
    DELETE FROM history_messages
    WHERE id BETWEEN (select maxid from max_id) - {n} AND (select maxid from max_id)
    ''')
    conn.commit()
    conn.close()


def add_filename_to_metadata(source, filename):
    with sqlite3.connect('metadata.db') as conn:
        conn.execute(f'''INSERT INTO uploaded_docs (global_source, filename) values ('{source}', '{filename}') ; ''')


def delete_filename_from_metadata(source, filename):
    with sqlite3.connect('metadata.db') as conn:
        conn.execute(f'''DELETE from uploaded_docs where global_source = '{source}' and filename ='{filename}' ; ''')



class ChromaManager:
    CHROMA_PATH_PRIMARY = f'./chroma/{current_user}/primary/'
    CHROMA_PATH_SECONDARY = f'./chroma/{current_user}/secondary/'
    USE_PRIMARY_CHROMA = True
    NEW_DATABASE_READY = False

    @classmethod
    def get_current_retriever(cls, embeddings):
        current_path = (
            cls.CHROMA_PATH_PRIMARY if cls.USE_PRIMARY_CHROMA
            else cls.CHROMA_PATH_SECONDARY
        )

        print(f"CURRENT CHROMA PATH: {current_path}")
        print(f"USE PRIMARY CHROMA: {cls.USE_PRIMARY_CHROMA}")

        # Каждый раз создаем новый retriever
        try:
            retriever = Chroma(
                persist_directory=current_path,
                embedding_function=embeddings
            ).as_retriever()

            # Проверка содержимого базы
            documents = retriever.get_relevant_documents("test")
            print(f"DOCUMENTS IN RETRIEVER: {len(documents)}")

            return retriever
        except Exception as e:
            print(f"ERROR CREATING RETRIEVER: {e}")
            return None