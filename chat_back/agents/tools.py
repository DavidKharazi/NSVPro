import os
import requests
from langchain_openai import ChatOpenAI
from authlib.integrations.starlette_client import OAuth
import time
import csv
import json
import re


os.environ['OPENAI_API_KEY'] = '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'


model_name = "gpt-4o-mini"
temperature = 0
llm = ChatOpenAI(model=model_name, temperature=temperature)
# embeddings = OpenAIEmbeddings()



oauth = OAuth()

def create_order(product_id: int, fio: str, phone: str, preorder: str) -> dict:
    """
    Создаёт заказ через API.

    Args:
        product_id (int): ID товара.
        fio (str): ФИО клиента.
        phone (str): Номер телефона в формате +375xxxxxxxxx.
        preorder (str): Статус предзаказа ('Y' или 'N').

    Returns:
        dict: Ответ от сервера с полями result и text.
    """
    url = "https://nsv.by/api/ai/"  # URL для отправки заказа

    payload = {
        "phone": phone,
        "fio": fio,
        "product_id": product_id,
        "ssid": "hCJOzTMiNXsqyIAyyg4a0bqBKCUNgQEm",
        "preorder": preorder
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "result": 0,
            "text": f"Ошибка при отправке запроса: {str(e)}"
        }


# Укажите URL для загрузки файла
CSV_URL = "https://nsv.by/dev/aicsv.php?key=cJT3qhgB9L9SjUmOko&ssid=1oAalTb506IyIonIFZLvgCa7LQ1VzsQQ&GEZhyH9Z=q0HR214dKNBFQHlHiIOZrlDxeCjyrmKWNbVhFqhZCPylzeRKHb"
# CSV_URL = "https://storage.yandexcloud.net/utlik/skycross/csv/aiCSV%20(6).csv"


# Укажите путь для папки, где будут храниться файлы
DATA_FOLDER = "data"
DATA_FOLDER_JSON = "upload_files"

# Укажите путь к вашему CSV файлу
csv_file_path = 'data/aiCSV.csv'

# Создание папок, если их нет
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER_JSON, exist_ok=True)


def download_csv():
    """Функция для загрузки файла CSV по URL, конвертация в JSON и сохранение для последующей загрузки в базу знаний."""
    try:
        """Сохранение нового файла"""
        save_path = os.path.join(DATA_FOLDER, f"aiCSV.csv")
        response = requests.get(CSV_URL)
        print(f"RESPONSE: {response}")
        response.raise_for_status()  # Проверка на ошибки HTTP

        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"Файл успешно загружен и сохранен в {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке файла: {e}")

    result = {}

    """Чтение CSV-файла и преобразование в структуру JSON"""
    with open(csv_file_path, encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        for row in reader:
            name = row['NAME']
            raw_price = row['PRICE']

            # Удаляем все символы, кроме цифр и точки
            cleaned_price = re.sub(r'[^\d.]', '', raw_price)
            price = float(cleaned_price) if cleaned_price else 0

            section_name = row['SECTION_NAME']

            # Проверка, начинается ли NAME со слова "Смартфон"
            if name.startswith("Смартфон"):
                if price > 3000:
                    section_name += ", дорогой телефон, крутой телефон, флагман, лучший, премиум, качественный, бомбический, мощный, кайфовый, стильный, трендовый"
                elif 1000 <= price <= 3000:
                    section_name += ", средний, обычный"
                elif price < 1000:
                    section_name += ", бюджетный, дешевый, простой, экономный, дешевый"

            # Добавляем запись в словарь, используя 'NAME' в качестве ключа
            result[name] = {
                "ID": row['ID'],
                "SECTION_NAME": section_name,
                "DESCRIPTION": row['DESCRIPTION'],
                "PRICE": row['PRICE'],
                "URL": row['URL'],
                "SALES": row['SALES'],
                "PREORDER": row['PREORDER']
            }

    """Удаление всех существующих файлов в папке DATA_FOLDER_JSON"""
    # time.sleep(2)
    for filename in os.listdir(DATA_FOLDER_JSON):
        file_path2 = os.path.join(DATA_FOLDER_JSON, filename)
        try:
            if os.path.isfile(file_path2):
                os.remove(file_path2)
                print(f"Удален файл: {file_path2}")
        except Exception as e:
            print(f"Не удалось удалить файл {file_path2}: {e}")

    """Сохранение результата в JSON файл"""
    # time.sleep(2)
    with open(f'upload_files/ai_nsv{time.process_time()}.json', 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

    print("Конвертация завершена. Данные сохранены в 'upload_files/ai_nsv*.json'.")

