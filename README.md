# Документация

## Описание

Чат с КС в виде виджета, состоит из 2х основных компонентов: chat-front и chat_back.

### Локальный запуск: 
1. Установить зависимости:
```bash
pip install -r requirements.txt
```
```bash
npm install
```
2. Выставить локальные пути:
#### перейти в директорию chat-front: 
chat-front/src/components/ChatBot/ChatBot.tsx в 52 и 193 строчке раскоментить локальный путь и закоментить веб.
выполнить:
```bash
npm run build
```
#### перейти в директорию chat_back: 
chat_back/app/main.py (строчка 79)
- раскоментировать: app.mount("/", StaticFiles(directory="../chat-front/dist", html=True), name="static")
- закоментировать: app.mount("/", StaticFiles(directory="./dist", html=True), name="static")

#### Важная информация перед запуском скрипта: 
- функция run_document_processing_cycle в chat_back/app/main.py отвечает за начало загрузки документов
- вызов download_csv() осуществляет загрузку при запуске скрипта
- при необходимости можно настроить периодическую зугрузку (по времени)
- после первого запуска необходимо дождаться загрузки документов (примерно 1 час)

#### Запуск скрипта: 
```bash
python -m app.main
```

#### Локальный доступ к приложению: http://localhost:8000/


### Веб запуск и деплой на Railway:

1. Выставить web пути:
#### перейти в директорию chat-front: 
chat-front/src/components/ChatBot/ChatBot.tsx в 52 и 193 строчке закоментить локальный путь и раскоментить веб.
выполнить:
```bash
npm run build
```
#### перейти в директорию chat_back: 
chat_back/app/main.py (строчка 79)
- закоментировать: app.mount("/", StaticFiles(directory="../chat-front/dist", html=True), name="static")
- раскоментировать: app.mount("/", StaticFiles(directory="./dist", html=True), name="static")

2. Сделать share project on github
3. Зарегистрироваться в Railway и синхронизировать со своим аккайунтом github:
- выбрать +create
- выбрать GITHUB Repo
- выбрать репозиторий проекта для деплоя
- дожаться концовки деплоя (10 минут) и зайти в Settings
- выбрать Generate domain и прописать тот веб путь, который указан в файле проекта: chat-front/src/components/ChatBot/ChatBot.tsx
- перейти по созданой ссылке.


