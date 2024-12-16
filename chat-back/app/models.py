import sqlite3

def init_metadata_db():
    with sqlite3.connect('metadata.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Admin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            ''')
        conn.execute('''
                CREATE TABLE IF NOT EXISTS uploaded_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                global_source TEXT,
                filename TEXT
                );
                ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255),
            name VARCHAR(255),
            cyberman_id INTEGER DEFAULT 1,
            chat_id INTEGER,
            is_active BOOLEAN DEFAULT FALSE,
            confirmation_token TEXT,
            reset_token TEXT,
            new_password VARCHAR(255),
            FOREIGN KEY (cyberman_id) REFERENCES Cyberman(id),
            FOREIGN KEY (chat_id) REFERENCES Chat(id)
        );
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS Cyberman (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255),
            creativity DOUBLE,
            prompt VARCHAR(255)
        );
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS Session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            user_id INTEGER,
            cyberman_id INTEGER,
            topic TEXT,
            FOREIGN KEY (user_id) REFERENCES Users(id),
            FOREIGN KEY (cyberman_id) REFERENCES Cyberman(id)
        );
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS Chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            messages VARCHAR(255),
            sender VARCHAR(255),
            sent_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
            session_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES Session(id)
        );
        ''')


init_metadata_db()
