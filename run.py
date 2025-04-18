from app import create_app

app = create_app()
app.config['SESSIONS_DICT'] = {}  # Инициализация словаря сессий

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6543)