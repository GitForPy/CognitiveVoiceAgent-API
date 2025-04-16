# settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Данные клиента 
    FULL_NAME: str = "Степанчук Алена Сергеевна"
    ACCOUNT_NUMBER: str = "111222333"
    DEBT_AMOUNT: str = "24 760 руб."
    ADDRESS: str = "Москва, Сокольнический вал, д. 20/3, кв. 3"

    # Данные компании
    COMPANY_NAME: str = "ЭнергоСбыт"
    COMPANY_PHONE: str = "8-900-555-55-55"

    # Сумма частичной оплаты (50% от DEBT_AMOUNT)
    PARTIAL_PAYMENT_AMOUNT: str = "12 380 руб."