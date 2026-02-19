# Управление сервисами Site-Img

## 📊 Статус сервисов

Проект состоит из двух сервисов, управляемых через systemd:

### 1. Frontend (Angular)
- **Порт**: 7070
- **Сервис**: `site-img-frontend.service`
- **URL**: http://109.172.115.171:7070

### 2. Backend (Python/FastAPI)
- **Порт**: 8000
- **Сервис**: `site-img-backend.service`
- **URL**: http://109.172.115.171:8000

## 🔧 Команды управления

### Проверка статуса
```bash
# Оба сервиса
systemctl status site-img-frontend.service site-img-backend.service

# Только frontend
systemctl status site-img-frontend.service

# Только backend
systemctl status site-img-backend.service
```

### Запуск/Остановка/Перезапуск
```bash
# Запуск
systemctl start site-img-frontend.service
systemctl start site-img-backend.service

# Остановка
systemctl stop site-img-frontend.service
systemctl stop site-img-backend.service

# Перезапуск
systemctl restart site-img-frontend.service
systemctl restart site-img-backend.service
```

### Включение/Отключение автозапуска
```bash
# Включить автозапуск при загрузке системы
systemctl enable site-img-frontend.service
systemctl enable site-img-backend.service

# Отключить автозапуск
systemctl disable site-img-frontend.service
systemctl disable site-img-backend.service
```

## 📝 Логи

### Просмотр логов
```bash
# Frontend логи
journalctl -u site-img-frontend.service -f

# Backend логи
journalctl -u site-img-backend.service -f

# Или файлы логов
tail -f /var/log/site-img-frontend.log
tail -f /var/log/site-img-backend.log
```

## 🔍 Проверка портов
```bash
# Проверить, что порты слушаются
ss -tlnp | grep -E ":(7070|8000)"

# Или
netstat -tlnp | grep -E ":(7070|8000)"
```

## 🚀 Автоматический запуск

Оба сервиса настроены на автоматический запуск при загрузке системы и автоматический перезапуск при сбоях:
- **Restart**: always
- **RestartSec**: 10 секунд

## ⚠️ Важные замечания

1. **Mediapipe**: Backend использует Dlib для обработки лиц, так как Mediapipe имеет проблемы совместимости с текущей версией TensorFlow
2. **PYTHONPATH**: Backend требует `PYTHONPATH=/root/site-img` для правильной работы импортов
3. **Порты**: Убедитесь, что порты 7070 и 8000 открыты в файрволе для внешнего доступа

## 🔥 Файрволл

Если сервисы не доступны извне, проверьте файрволл:
```bash
# UFW
sudo ufw allow 7070
sudo ufw allow 8000

# iptables
sudo iptables -A INPUT -p tcp --dport 7070 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

## 📍 Расположение файлов

- **Systemd сервисы**: `/etc/systemd/system/site-img-*.service`
- **Код проекта**: `/root/site-img`
- **Логи**: `/var/log/site-img-*.log`
