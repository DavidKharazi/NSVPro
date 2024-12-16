//
//
// import React from 'react';
// import styles from './ChatMessage.module.scss';
// import { IChatMessage } from '../../types/Chat';
// import smallIcon from '../ChatBot/assets/icons/man.jpg';
//
// // Функция для форматирования сообщения
// function formatMessage(text: string) {
//   // const phoneRegex = /(\+?\d{1,3}\s?\(?\d{1,4}\)?\s?\d{1,3}[\s.-]?\d{1,2}[\s.-]?\d{1,2}[\s.-]?\d{1,2})/g;
//   const phoneRegex = /(\+?375[\s-]?\(?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})/g;
//   const urlRegex = /(https?:\/\/[^\s)]+[^\s.,)])/g;
//   const boldTextRegex = /\*\*([^*]+)\*\*/g;
//   const squareBracketRegex = /\[[^\]]*\]/g;
//   const hashRegex = /^#{2,}\s+/;
//
//   return (
//     <>
//
//       {text.split('\n').map((paragraph, index) => {
//         // Удаляем текст в квадратных скобках
//         // const cleanedText = paragraph.replace(squareBracketRegex, '');
//         const cleanedText = paragraph.replace(hashRegex, '').replace(squareBracketRegex, '');
//
//
//         // Заменяем выделение жирным через `**`
//         const formattedParagraph = cleanedText.replace(boldTextRegex, '<strong>$1</strong>');
//
//         const formattedText = formattedParagraph.split(urlRegex).map((part, idx) => {
//           if (urlRegex.test(part)) {
//             return (
//               <button
//                 key={idx}
//                 onClick={() => window.open(part, '_blank')}
//                 style={{
//                   background: 'none',
//                   border: 'none',
//                   color: 'blue',
//                   cursor: 'pointer',
//                   textDecoration: 'underline'
//                 }}
//               >
//                 Подробнее✅
//               </button>
//             );
//           } else {
//             return part.split(phoneRegex).map((subPart, subIdx) => {
//               if (phoneRegex.test(subPart)) {
//                 return (
//                   <a
//                     key={subIdx}
//                     href={`tel:${subPart.replace(/\s+/g, '')}`}
//                     style={{ color: 'blue', textDecoration: 'underline' }}
//                   >
//                     {subPart}
//                   </a>
//                 );
//               }
//               return (
//                 <span
//                   key={subIdx}
//                   dangerouslySetInnerHTML={{ __html: subPart }}
//                 />
//               );
//             });
//           }
//         });
//
//         const isListItem = cleanedText.match(/^(\d+\.|-)\s/);
//         const tag = isListItem ? 'li' : 'p';
//
//         return (
//           <div key={index} style={{ margin: '10px 0' }}>
//             {React.createElement(tag, {}, formattedText)}
//           </div>
//         );
//       })}
//     </>
//   );
// }
//
// // Компонент ChatMessage
// function ChatMessage({ text, isMe }: IChatMessage) {
//   return (
//     <>
//       {isMe ? (
//         <div className={styles.reqAi}>{formatMessage(text)}</div>
//       ) : (
//         <div className={styles.chatMessagesWrapper}>
//           <div className={styles.chatMessagesIcon}>
//             <img src={smallIcon} alt="support-small-image" width="40" height="40" />
//           </div>
//           <div className={styles.resAi}>{formatMessage(text)}</div>
//         </div>
//       )}
//     </>
//   );
// }
//
// export default ChatMessage;



import React from 'react';
import styles from './ChatMessage.module.scss';
import { IChatMessage } from '../../types/Chat';
import smallIcon from '../ChatBot/assets/icons/man.jpg';

// Функция для форматирования сообщения
function formatMessage(text: string) {
  const phoneRegex = /(\+?375[\s-]?\(?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})/g;
  const urlRegex = /(https?:\/\/[^\s)]+[^\s.,)])/g;
  const boldTextRegex = /\*\*([^*]+)\*\*/g;
  const squareBracketRegex = /\[[^\]]*\]/g;
  const hashRegex = /^#{2,}\s+/;
  // Новый регекс для поиска ID: с последующим текстом
  // const idRegex = /(ID:\s*[^\n]+)/gi;
  const idRegex = /(ID:\s*[^ ]+ [^\s]+)(.*)/gi;

  return (
    <>
      {text.split('\n').map((paragraph, index) => {
        const cleanedText = paragraph.replace(hashRegex, '').replace(squareBracketRegex, '');

        // Обработка ID: формата
        let formattedParagraph = cleanedText.replace(idRegex, (match) => {
          return `<span style="font-size: 0.8em; text-transform: lowercase;">${match}</span>`;
        });

        // Обработка жирного текста
        formattedParagraph = formattedParagraph.replace(boldTextRegex, '<strong>$1</strong>');

        const formattedText = formattedParagraph.split(urlRegex).map((part, idx) => {
          if (urlRegex.test(part)) {
            return (
              <button
                key={idx}
                onClick={() => window.open(part, '_blank')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'blue',
                  cursor: 'pointer',
                  textDecoration: 'underline'
                }}
              >
                Подробнее✅
              </button>
            );
          } else {
            return part.split(phoneRegex).map((subPart, subIdx) => {
              if (phoneRegex.test(subPart)) {
                return (
                  <a
                    key={subIdx}
                    href={`tel:${subPart.replace(/\s+/g, '')}`}
                    style={{ color: 'blue', textDecoration: 'underline' }}
                  >
                    {subPart}
                  </a>
                );
              }
              return (
                <span
                  key={subIdx}
                  dangerouslySetInnerHTML={{ __html: subPart }}
                />
              );
            });
          }
        });

        const isListItem = cleanedText.match(/^(\d+\.|-)\s/);
        const tag = isListItem ? 'li' : 'p';
        return (
          <div key={index} style={{ margin: '10px 0' }}>
            {React.createElement(tag, {}, formattedText)}
          </div>
        );
      })}
    </>
  );
}

// Компонент ChatMessage
function ChatMessage({ text, isMe }: IChatMessage) {
  return (
    <>
      {isMe ? (
        <div className={styles.reqAi}>{formatMessage(text)}</div>
      ) : (
        <div className={styles.chatMessagesWrapper}>
          <div className={styles.chatMessagesIcon}>
            <img src={smallIcon} alt="support-small-image" width="40" height="40" />
          </div>
          <div className={styles.resAi}>{formatMessage(text)}</div>
        </div>
      )}
    </>
  );
}

export default ChatMessage;