import os
import time
import requests
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Настройки
# =========================

MODEL_NAME = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

MAX_RUNTIME = int(os.getenv("MAX_RUNTIME_SECONDS", "20700"))

START_TIME = time.time()

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# =========================
# Загрузка модели
# =========================

print(f"Load Model {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully.")


# =========================
# История диалогов
# =========================

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Answer the user clearly and directly."
        )
    }
]


# =========================
# Генерация ответа
# =========================

def generate_answer(user_text):

    global messages

    messages.append({
        "role": "user",
        "content": user_text
    })

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    new_tokens = outputs[0][inputs.shape[-1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# =========================
# Telegram
# =========================

def telegram(method, **params):

    url = f"{TELEGRAM_API}/{method}"

    response = requests.post(
        url,
        json=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def send_message(chat_id, text):

    # Telegram ограничивает сообщение примерно 4096 символами.
    # Разбиваем длинные ответы.

    for i in range(0, len(text), 4000):

        telegram(
            "sendMessage",
            chat_id=chat_id,
            text=text[i:i + 4000]
        )


# =========================
# Основной цикл
# =========================

def main():

    print("Telegram AI bot started.")

    offset = None

    while True:

        # Выходим через 5 часов 45 минут
        if time.time() - START_TIME >= MAX_RUNTIME:

            print("Maximum runtime reached. Exiting...")

            break

        try:

            params = {
                "timeout": 50
            }

            if offset is not None:
                params["offset"] = offset

            result = telegram(
                "getUpdates",
                **params
            )

            if not result.get("ok"):
                print("Telegram API error:", result)
                time.sleep(5)
                continue

            updates = result.get("result", [])

            for update in updates:

                offset = update["update_id"] + 1

                message = update.get("message")

                if not message:
                    continue

                chat = message.get("chat", {})
                chat_id = chat.get("id")

                text = message.get("text")

                if not text:
                    continue

                print(
                    f"Message from {chat_id}: {text}"
                )

                if text == "/start":

                    send_message(
                        chat_id,
                        "🤖 AI запущен.\n"
                        "Напиши мне сообщение."
                    )

                    continue

                if text == "/status":

                    elapsed = int(time.time() - START_TIME)
                    remaining = max(0, MAX_RUNTIME - elapsed)

                    minutes = remaining // 60

                    send_message(
                        chat_id,
                        f"🟢 AI работает.\n"
                        f"Осталось примерно {minutes} минут."
                    )

                    continue

                send_message(
                    chat_id,
                    "⏳ Думаю..."
                )

                try:

                    answer = generate_answer(text)

                    send_message(
                        chat_id,
                        answer
                    )

                except Exception as e:

                    print(
                        "Generation error:",
                        repr(e)
                    )

                    send_message(
                        chat_id,
                        "❌ Ошибка при генерации ответа."
                    )

        except Exception as e:

            print(
                "Telegram error:",
                repr(e)
            )

            time.sleep(10)


if __name__ == "__main__":
    main()
