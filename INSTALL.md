# Инструкция по развертыванию проекта на новом сервере

### 1. Клонирование репозитория
```bash
git clone https://github.com/kumyk95/img.git
cd img
```

### 2. Загрузка весов моделей (обязательно)
Модели весят ~800мб и скачиваются отдельно:
```bash
chmod +x download_models.sh
./download_models.sh
```

### 3. Установка системных зависимостей
Необходимы для работы нейросетей (OpenCV, Dlib):
```bash
apt update
apt install -y cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-pip nodejs npm
```

### 4. Установка зависимостей проекта
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
npm install
```

### 5. Настройка автозапуска (Systemd)
Если проект лежит не в `/root/site-img`, откройте файлы `.service` и исправьте пути в строках `WorkingDirectory` и `ExecStart`.

Затем запустите службы:
```bash
cp site-img-*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now site-img-frontend site-img-backend
```

### 6. Команды управления
*   **Статус:** `systemctl status site-img-backend`
*   **Логи:** `journalctl -u site-img-backend -f`
*   **Перезапуск:** `systemctl restart site-img-backend`
