
import { useState, useEffect, useRef, KeyboardEvent } from 'react';
import { toast } from 'react-toastify';
import styles from './ChatBot.module.scss';
import microOff from './assets/icons/microOff.svg';
import icon from './assets/icons/man.jpg';
import ChatMessage from '../ChatMessage/ChatMessage';
import { IChatBotProps, IChatMessage } from '../../types/Chat';
import SpeechRecognition, {
  useSpeechRecognition,
} from 'react-speech-recognition';

function ChatBot({ closeChat, isLeft }: IChatBotProps) {
  const {transcript, listening, resetTranscript} = useSpeechRecognition();
  const [message, setMessage] = useState<IChatMessage>({
    text: '',
    isMe: true,
  });
  const [chat, setChat] = useState<IChatMessage[]>([
    {text: 'Введите ваши данные для начала чата.', isMe: false},
  ]);
  const [isWriting, setIsWriting] = useState(false);
  const [isRegistered, setIsRegistered] = useState(false); // Флаг регистрации
  const [email, setEmail] = useState(''); // Поле email
  const [phone, setName] = useState(''); // Поле name
  const [, setSocket] = useState<WebSocket | null>(null); // WebSocket

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inpTextRef = useRef<HTMLDivElement>(null);
  // @ts-ignore
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null); // Используем useRef для хранения WebSocket

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
          chatContainerRef.current.scrollHeight;
    }
  }, [chat]);

  useEffect(() => {
    if (localStorage.getItem('username')) {
      setIsRegistered(true);
    }
  }, []);

  useEffect(() => {
    const initializeWebSocket = () => {
      const username = localStorage.getItem('username');
      if (username && !wsRef.current) {
        // Проверяем, что WebSocket еще не создан
        const ws = new WebSocket(`ws://localhost:8000/ws/rag_chat/?email=${encodeURIComponent(username)}`);
        // const ws = new WebSocket(
        //     `wss://nsvcyberman.up.railway.app/ws/rag_chat/?email=${encodeURIComponent(
        //         username
        //     )}`
        // );
        wsRef.current = ws; // Сохраняем WebSocket в useRef

        ws.onopen = () => {
          console.log('WebSocket connection established');
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.answer) {
            setChat((prevChat) => [
              ...prevChat,
              {text: data.answer, isMe: false},
            ]);
            setIsWriting(false);
          }
        };

        ws.onclose = async () => {
          console.log('WebSocket connection closed');
          if (ws.readyState === WebSocket.CLOSED) {
            try {
              await sendChatHistory(); // Дождаться отправки истории чата перед закрытием соединения
            } catch (error) {
              console.error('Error sending chat history:', error);
            }
          }
        };

        ws.onerror = (error) => console.error('WebSocket error:', error);
      }
    };

    initializeWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close(); // Закрытие WebSocket соединения
        wsRef.current = null; // Очищаем ссылку на WebSocket
      }
    };
  }, []); // Пустой массив зависимостей, чтобы выполнить этот эффект только один раз

  useEffect(() => {
    const loadChatMessages = async () => {
      const chatId = localStorage.getItem('ChatId');
      if (chatId) {
        try {
          const response = await fetch(
              `/get_chat_messages/${chatId}`
          );
          if (!response.ok) {
            throw new Error('Не удалось загрузить сообщения чата');
          }
          const data = await response.json();
          setChat(
              data.messages.map((msg: { messages: string }) => ({
                text: msg.messages,
                isMe: false,
              }))
          );
        } catch (error) {
          console.error(error);
        }
      }
    };

    if (isRegistered) {
      loadChatMessages();
    }
  }, [isRegistered]);

  useEffect(() => {
    if (transcript) {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      const id = setTimeout(() => {
        sendMessageToBot(transcript);
        resetTranscript(); // Сброс текста после отправки
      }, 1000);

      setTimeoutId(id);
    }
  }, [transcript]);

  const handleRegister = async () => {
    if (!email || !phone) {
      toast.warn('Введите имя и телефон');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('user', email);
      formData.append('phone', phone);
      const response = await fetch(
          '/register/',
          {
            method: 'POST',
            body: formData,
          }
      );

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Ошибка регистрации');
      }

      localStorage.setItem('username', phone);
      localStorage.setItem('ChatId', data.session_id);
      localStorage.setItem('name', email);

      setIsRegistered(true);

      const username = localStorage.getItem('username') || '';
      if (username) {
        const response = await fetch(
            '/create_new_chat/',
            {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({email: username}),
            }
        );

        if (response.ok) {
          const data = await response.json();
          localStorage.setItem('ChatId', data.session_id); // Сохраняем session_id
        } else {
          throw new Error('Не удалось создать новый чат');
        }
      }


      const ws = new WebSocket(`ws://localhost:8000/ws/rag_chat/?email=${encodeURIComponent(username)}`);
      // const ws = new WebSocket(
      //     `wss://nsvcyberman.up.railway.app/ws/rag_chat/?email=${encodeURIComponent(
      //         username
      //     )}`
      // );
      setSocket(ws);
      wsRef.current = ws; // Сохраняем WebSocket в useRef

      ws.onopen = () => {
        setChat((prevChat) => [
          ...prevChat,
          {text: data.message, isMe: false},
        ]);
        setIsRegistered(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.answer) {
          setChat((prevChat) => [
            ...prevChat,
            {text: data.answer, isMe: false},
          ]);
          setIsWriting(false);
        }
      };

      ws.onclose = async () => {
        console.log('WebSocket connection closed');
        if (ws.readyState === WebSocket.CLOSED) {
          try {
            await sendChatHistory(); // Дождаться отправки истории чата перед закрытием соединения
          } catch (error) {
            console.error('Error sending chat history:', error);
          }
        }
      };
      ws.onerror = (error) => console.error('Ошибка WebSocket:', error);
    } catch (error) {
      toast.error(
          error instanceof Error ? error.message : 'Произошла неизвестная ошибка'
      );
      setChat((prevChat) => [
        ...prevChat,
        {
          text: 'Пользователь с данным номером телефона уже есть.',
          isMe: false,
        },
      ]);
    }
  };

  useEffect(() => {
    if (listening) {
      setMessage((prevMessage) => ({...prevMessage, text: transcript}));

      const timeoutId = setTimeout(() => {
        SpeechRecognition.stopListening();
      }, 7000);

      return () => clearTimeout(timeoutId);
    }
  }, [transcript, listening]);

  const handleRecordVoice = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      SpeechRecognition.startListening({continuous: true});
    }
  };

  const handleInput = () => {
    if (inpTextRef.current) {
      const text = inpTextRef.current.innerText;
      setMessage({text, isMe: true});
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!isWriting && e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessageToBot(message.text);
    }
  };

  const sendMessageToBot = async (text: string) => {
    if (!text.trim()) {
      toast.warn('Пожалуйста, напишите сообщение');
      return;
    }

    const sessionId = localStorage.getItem('ChatId');
    if (!sessionId) {
      toast.error('Не удалось найти session_id');
      return;
    }

    if (chat.length > 0 && chat[chat.length - 1].text === text.trim()) {
      toast.warn('Это сообщение уже было отправлено');
      return;
    }

    setIsWriting(true);
    setChat([...chat, {text, isMe: true}]);
    setMessage({text: ''});
    if (inpTextRef.current) {
      inpTextRef.current.innerText = '';
    }

    if (wsRef.current) {
      wsRef.current.send(
          JSON.stringify({
            question_data: {question: text, session_id: sessionId},
          })
      );
    }

    SpeechRecognition.stopListening();
    resetTranscript();
  };

  const sendChatHistory = () => {
    const sessionId = localStorage.getItem('ChatId');
    const username = localStorage.getItem('username');
    const name = localStorage.getItem('name');

    if (!sessionId) {
      console.error('Session ID not found in localStorage');
      return;
    }

    const apiUrl =
        '/save_chat_history/' + sessionId;

    const data = {
      username: username,
      name: name,
    };

    const blob = new Blob([JSON.stringify(data)], {type: 'application/json'});
    navigator.sendBeacon(apiUrl, blob);
  };


  return (
      <div className={styles.botContainer}>
        <div className={styles.botWrapper}>
          <div className={styles.botHeader}>
            <button type="button" className={styles.botHeaderArrow}></button>
            Олег
          </div>
          <div className={styles.chat} ref={chatContainerRef}>
            <div className={styles.chatIcon}>
              <div className={styles.chatImg}>
                <img src={icon} alt="support-image" width="70" height="70"/>
              </div>
              <span className={styles.chatSupportTime}>Всегда в сети</span>
              <p className={styles.chatSupport}>Support</p>
            </div>
            <div className={styles.chatMessages}>
              {chat.map((msg, index) => (
                  <ChatMessage key={index} text={msg.text} isMe={msg.isMe}/>
              ))}
            </div>
            {isWriting && (
                <div className={styles.botWriting}>
                  <p>Олег пишет...</p>
                  <span className={styles.loader}></span>
                </div>
            )}
          </div>

          {!isRegistered ? (
              <div className={styles.registerForm}>
                <input
                    type="text"
                    placeholder="Введите имя"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                />
                <div className={styles.phoneInput}>
                  <span>+375</span>
                  {/* Префикс телефона */}
                  <input
                      type="text"
                      maxLength={9} // Ограничиваем ввод 9 цифрами
                      placeholder="9 цифр"
                      value={phone}
                      onChange={(e) => {
                        const value = e.target.value;
                        if (/^\d*$/.test(value)) {
                          // Проверяем, что вводятся только цифры
                          setName(value);
                        }
                      }}
                  />
                </div>
                <button onClick={handleRegister}>Отправить</button>
              </div>
          ) : (
              <div className={styles.inpForm}>
                <div
                    className={styles.inpText}
                    contentEditable
                    role="textbox"
                    onKeyDown={handleKeyDown}
                    onInput={handleInput}
                    ref={inpTextRef}
                    data-placeholder="Введите ваше сообщение"
                />
                <div className={styles.btnsWrapper}>
                  <button
                      disabled={isWriting}
                      className={listening ? styles.recording : styles.btnVoice}
                      onClick={handleRecordVoice}
                  >
                    {!listening && <img src={microOff} alt="micro-icon"/>}
                  </button>
                  <button
                      disabled={isWriting}
                      className={
                        message.text.length
                            ? styles.inpSubmitActive
                            : isWriting
                                ? styles.inpSubmitStop
                                : styles.inpSubmit
                      }
                      onClick={() => sendMessageToBot(message.text)}
                  />
                </div>
              </div>
          )}

          {/* Надпись под текстовым вводом */}
          <span className={styles.inputBottomText}>
            Powered by
            <a
                href="https://www.linkedin.com/company/utlik-co/mycompany/"
                target="_blank"
                rel="noopener noreferrer"
            >
              &nbsp;Utlik.
            </a>
          </span>

        </div>
        <button
            className={isLeft ? styles.closeBtnRight : styles.closeBtnLeft}
            onClick={() => closeChat()}
        ></button>
      </div>
  );
}

export default ChatBot;
