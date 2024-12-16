import { useState } from 'react';
import classNames from 'classnames';
import ChatBot from '../ChatBot/ChatBot';
import styles from './ChatWidget.module.scss';
import { IChatWidgetProps } from '../../types/Chat';
import { toast } from 'react-toastify';

function ChatWidget({ isLeft = false }: IChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  const checkUserAndOpenChat = async () => {
    const username = localStorage.getItem('username');
    if (username) {
      const response = await fetch(
        '/create_new_chat/',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: username }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('ChatId', data.session_id); // Сохраняем session_id
      } else if (response.status === 404) {
        // Если получен 404, не удалось создать новый чат — открываем форму регистрации
        localStorage.removeItem('username'); // Удаляем username, чтобы отобразить форму
      } else {
        toast.error('Не удалось создать новый чат');
        return;
      }
    }
    setIsOpen(true); // Открываем чат
  };

  const handleVisibility = () => {
    if (isOpen) {
      setIsExiting(true);
      setTimeout(() => {
        setIsOpen(false);
        setIsExiting(false);
      }, 200);
    } else {
      checkUserAndOpenChat();
    }
  };

  return (
    <div className={styles.widgetOverlay}>
      <div
        className={classNames(styles.widgetContainer, {
          [styles.widgetContainerIn]: !isExiting,
          [styles.widgetContainerOut]: isExiting,
          [styles.widgetContainerRight]: !isLeft,
        })}
      >
        {isOpen && (
          <div
            className={classNames({
              [styles.widgetChat]: !isExiting,
              [styles.widgetChatExiting]: isExiting,
            })}
          >
            <ChatBot closeChat={handleVisibility} isLeft={isLeft} />
          </div>
        )}
        {!isOpen && (
          <button className={styles.widgetButton} onClick={handleVisibility}>
          </button>
        )}
      </div>
    </div>
  );
}

export default ChatWidget;