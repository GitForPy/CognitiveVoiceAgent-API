import requests

BASE_URL = "http://77.239.108.6:6543"

# Получение chat_id (передаем непустой JSON, например {"dummy": true})
response = requests.get(f"{BASE_URL}/api/get_new_chat_id", json={"dummy": True})
data = response.json()
print("Полученный ответ:", data)
chat_id = data["chat_id"]
print("Получен chat_id:", chat_id)

chat_id = 'b24aceb2-2757-4337-a9b8-85e03f4908da'


# # Отправка ответа "Да" для перехода в следующую фазу (идентификация -> debt_discussion)
# response = requests.post(
#     f"{BASE_URL}/api/chat",
#     json={"chat_id": chat_id, "message": "не планирую"}
# )
# data = response.json()
# print("Ответ бота после подтверждения:", data)

# Отправка сообщения для обсуждения задолженности
# response = requests.post(
#     f"{BASE_URL}/start_chat",
#     json={"chat_id": chat_id, "message": "Привет, когда я могу оплатить?"}
# )
# data = response.json()
# print("Ответ бота в фазе задолженности:", data)

# Принудительное удаление диалога (при необходимости)
# response = requests.post(f"{BASE_URL}/remove_chat", json={"chat_id": chat_id})
# print("Удаление чата:", response.json())
