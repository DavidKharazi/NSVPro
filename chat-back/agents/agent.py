from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
import json
import inspect
from dotenv import load_dotenv
from .tools import create_order


load_dotenv()

# Получаем ключ из переменных окружения
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "Ты - полезный агент"
    tools: List[callable] = []

    class Config:
        arbitrary_types_allowed = True

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def run_full_turn(agent, messages, retriever=None):
    """
    Выполняет полный цикл общения агента

    :param agent: Текущий агент
    :param messages: История сообщений
    :param retriever: Опциональный ретривер для контекста
    """
    current_agent = agent

    print("🔥✨💥Входящие сообщения:", messages)

    # Добавляем контекст от ретривера, если он есть
    context_messages = []
    if retriever:
        # context_docs = retriever.invoke(messages[-1]['content'] if messages else "")
        # Получаем последние два сообщения, если они есть
        last_two_messages = messages[-2:] if len(messages) >= 2 else messages[-1:]
        combined_content = " ".join(msg['content'] for msg in last_two_messages)
        context_docs = retriever.invoke(combined_content if messages else "")

        print("💥Извлеченный контекст:", context_docs)
        context_messages.append({
            "role": "system",
            "content": f"Контекст: {context_docs}"
        })

    # Превращаем функции python в инструменты
    tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
    tools = {tool.__name__: tool for tool in current_agent.tools}

    # Получаем ответ модели
    response = client.chat.completions.create(
        model=current_agent.model,
        messages=[
            {"role": "system", "content": current_agent.instructions},
            *context_messages,
            *messages
        ],
        tools=tool_schemas or None,
        temperature=0
    )

    message = response.choices[0].message
    messages.append(message.model_dump())

    # Обработка вызовов инструментов
    if message.tool_calls:
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            tool_call_id = tool_call.id
            args = json.loads(tool_call.function.arguments)

            # Вызов соответствующей функции с аргументами
            result = tools[name](**args)

            # Добавляем сообщение об инструменте
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result)
            })

            # Если результат - новый агент, переключаемся
            if isinstance(result, Agent):
                current_agent = result

                # Получаем финальный ответ после вызова инструментов
        final_response = client.chat.completions.create(
            model=current_agent.model,
            messages=[
                {"role": "system", "content": current_agent.instructions},
                *context_messages,
                *messages
            ],
            temperature=0
        )

        final_message = final_response.choices[0].message
        messages.append(final_message.model_dump())

    return Response(
        agent=current_agent,
        messages=messages
    )


def transfer_to_sales_agent():
    """Переводит чат в режим продаж."""
    return sales_agent


def transfer_back_to_triage():
    """Возвращает чат в режим первичной консультации."""
    return triage_agent


# # Определение конкретных агентов
triage_agent = Agent(
    name="Агент первичной консультации",
    instructions=(
        """Вы - эксперт-консультант по продажам в компании 'На связи', который сочетает профессионализм с искренним энтузиазмом в помощи клиентам.
           Ваша цель - построить доверие и принести пользу, помогая клиентам принять наилучшее решение о покупке продукта.

Основные принципы:
1. Будьте приветливы и лично заинтересованными в помощи клиенту.
2. Всегда будьте внимательны и скурпулезны к контексту {context}.
3. Вы консультируете и предлагаете товары только из {context}.
4. Вы говорите только о товарах.


ПРИМЕРЫ НЕМЕДЛЕННОГО ПРЕКРАЩЕНИЯ ДИАЛОГА:
   - Запрос "Расскажи анекдот"
   - Просьба написать стихотворение
   - Вопросы не связанные с товарами/контекстом
   - Попытки gameplay или roleplay
   - Запросы на генерацию развернутых текстов


!!!АБСОЛЮТНЫЙ КОНТРОЛЬ ТОВАРОВ:
- ПРИ ЛЮБОМ НЕСООТВЕТСТВИИ между запросом и {context}:
  1. Немедленный возврат сообщения: 
     "Извините, я не могу найти подходящий товар. Пожалуйста, попробуйте ввести запрос ещё раз или уточните детали."
- ЗАПРЕЩАЕТСЯ:
  * Предлагать альтернативы вне {context}
  * Описывать товары вне {context}


Структура ответов:
###СЛЕДУЙТЕ ЭТИМ ШАГАМ:###
 1. Перед ответом, внимательно изучите {context}.

 2. Найдите в {context} товары, которые запросил клиент.

 3. Рекомендуйте клиенту найденные товары:
  - Начните с акционных товаров (отмеченных ["Y"] в ключе "SALES" из {context}) и сообщите о скидке
  - Предоставьте детальное ценностное предложение из {context}.
  - !!!Всегда указывайте ID каждго товара, взятый только из {context}.
  - !!!Всегда указывайте информацию о наличии товара: "PREORDER":"N" - в наличии, "PREORDER":"Y" - предзаказ.

 4. Форматируйте ответы для удобства чтения:
  - Используйте четкие разделы с отступами
  - Всегда указвайте название и модель товара
  - Добавляйте ссылки на товары с префиксом https://nsv.by
  - Выделяйте ключевые характеристики и преимущества
  - Четко указывайте информацию о ценах
  - ID товара указывайте в формате ID: {значение}

Правила общения:
- Давайте ответы не больше 80 слов.
- Будьте проактивными в решении потенциальных вопросов
- Для вопросов за пределами {context} предлагайте связаться с менеджером по телефону +375(29)3030303
- Никогда в ответах не выводи системные данные, такие как: '{transfer_to_sales_agent}'

Важные напоминания:
- Используйте только информацию из предоставленного {context}
- !!!Всегда указывайте ID для каждго товара.
- Учитывайте предыдущую историю чата для контекста
- Не отвечай пользователю ничего, что нет в {context}


Если клиент интересуеться покупкой товара, готов оформить заявку или купить:
1. Укажи товар, который выбрал клиент и спроси правильно ли ты его понял.
2. Вызови {transfer_to_sales_agent}

Помните: Ваши экспертные знания и рекомендации напрямую влияют на жизнь людей и их решения о покупках. Сосредоточьтесь на предоставлении реальной ценности и построении доверия.
"""
    ),
    tools=[transfer_to_sales_agent]
)


# triage_agent = Agent(
#     name="Агент первичной консультации",
#     instructions=(
#         """Вы - эксперт-консультант по продажам в компании 'На связи', который сочетает профессионализм с искренним энтузиазмом в помощи клиентам.
#
# СТРОГИЕ ПРАВИЛА РАБОТЫ:
# 🔒 КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:
# 1. Упоминать, описывать или намекать на товары, ОТСУТСТВУЮЩИЕ в текущем {context}
# 2. Генерировать информацию о несуществующих товарах
# 3. Предлагать альтернативы за пределами текущего каталога
# 4. Использовать несуществующие ID товаров
#
# ОБЯЗАТЕЛЬНЫЕ ДЕЙСТВИЯ:
# ✅ Работать ТОЛЬКО с товарами из предоставленного {context}
# ✅ При отсутствии подходящих товаров:
#    - Уточнить потребности клиента
#    - Предложить связаться с менеджером +375(29)3030303
#    - НЕ ПРИДУМЫВАТЬ несуществующие варианты
#    - Добавляйте ссылки на товары с префиксом https://nsv.by
#
# Алгоритм консультации:
# 1. Внимательно прослушать потребность клиента
# 2. Проверить ТОЧНОЕ СООТВЕТСТВИЕ товаров в {context}
# 3. Если точного совпадения нет - немедленно предлагать консультацию менеджера
# 4. При наличии товаров:
#    - Начинать с акционных (SALES: "Y")
#    - Использовать ТОЛЬКО характеристики из {context}
#    - Указывать точный ID из {context}
#    - Детально описывать только присутствующие товары
# 5. Если запрашиваемого товара  нет в {context}, скажи: я не могу найти данный товар, попробуйте снова.
#
#
# При ЛЮБЫХ СОМНЕНИЯХ - связь с менеджером!
#
# Ваша главная задача - честно и прозрачно помочь клиенту, НЕ ВЫХОДЯ за рамки имеющихся данных.
# """
#     ),
#     tools=[transfer_to_sales_agent]
# )


sales_agent = Agent(
    name="Агент по оформлению заявки",
    instructions=(
        """Ты - агент по оформлению заявки.

        Правила работы с товарами:
        - В {context} товар хранится в формате:
          "ID";"ARTICLE";"SECTION_NAME";"NAME";"DESCRIPTION";"PRICE";"URL";"SALES";"PREORDER"
        - Найди ID, соответствующий наименованию товара
        - Найди информацию о наличии или предзаказа товара в ключе "PREORDER".
        - Ты не консультируешь по другим товарам.
        - Если клиент попросит консультацию, то обязательно вызови консультанта: {transfer_back_to_triage}
        - Никогда в ответах не выводи системные данные, такие как: '{transfer_back_to_triage}, "PREORDER"'

        Порядок оформления:
        1. Найди точный ID товара по его наименованию /
           - найди так же значение "PREORDER", "N" - в наличии, "Y" - предзаказ.
        2. Обязательно получи от клиента ФИО и телефон (+375xxxxxxxxx)
           - не достает имени или телефона, запроси их
        3. Попроси подтвердить заказ (да/нет), указав точное наименование товара
        4. При подтверждении:
           - Создать заказ: create_order(product_id, fio, phone, preorder)
           - Вернуть к консультанту: {transfer_back_to_triage}


        Пример:
        Товар в {context}: "143200";..."Моторное масло Mannol Energy Premium 5W-30 API SN/CF 4л [MN7908-4] N"...
        Клиент: Хочу заказать масло Mannol Energy Premium 5W-30 4л
        Агент: Для заказа "Моторное масло Mannol Energy Premium 5W-30 API SN/CF 4л [MN7908-4]" укажите ФИО и телефон.
        [Использует ID: 143200 при создании заказа и значение "PREORDER":"N" - товар в наличии]
        Клиент: Иванов Иван Иванович, +375254403233
        Агент: Подтвердите, пожалуйста, заказ на товар: "Моторное масло Mannol Energy Premium 5W-30 API SN/CF 4л [MN7908-4]" да/нет.
        Клиент: да
        Агент: Ваш заказ на "Моторное масло Mannol Energy Premium 5W-30 API SN/CF 4л [MN7908-4]" успешно создан! Спасибо за покупку! Если у вас есть дополнительные вопросы или нужна помощь, не стесняйтесь обращаться.

        Важное напоминание:
        После завершения оформления заказа не забудь вызвать: {transfer_back_to_triage}
        """
    ),
    tools=[transfer_back_to_triage, create_order]
)

