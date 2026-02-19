#!/bin/bash

# Скрипт автоматического развертывания проекта site-img
# Использование: bash deploy.sh

set -e  # Остановить выполнение при ошибке

echo "========================================="
echo "  Развертывание проекта site-img"
echo "========================================="
echo ""

# Проверка наличия Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js не установлен!"
    echo "Установка Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

echo "✓ Node.js версия: $(node -v)"
echo "✓ npm версия: $(npm -v)"
echo ""

# Проверка наличия Angular CLI
if ! command -v ng &> /dev/null; then
    echo "Установка Angular CLI..."
    npm install -g @angular/cli@21.0.2
fi

echo "✓ Angular CLI установлен"
echo ""

# Установка зависимостей
echo "Установка зависимостей проекта..."
npm install

echo ""
echo "✓ Зависимости установлены"
echo ""

# Остановка предыдущего процесса на порту 7070 (если есть)
echo "Проверка порта 7070..."
if lsof -Pi :7070 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Остановка процесса на порту 7070..."
    kill -9 $(lsof -t -i:7070) 2>/dev/null || true
fi

echo ""
echo "========================================="
echo "  Запуск проекта на порту 7070"
echo "========================================="
echo ""

# Запуск в фоновом режиме
nohup ng serve --host 0.0.0.0 --port 7070 > /var/log/site-img.log 2>&1 &

echo "✓ Проект запущен на порту 7070"
echo ""
echo "Проверьте доступность: http://$(hostname -I | awk '{print $1}'):7070"
echo "Логи доступны в: /var/log/site-img.log"
echo ""
echo "Для просмотра логов: tail -f /var/log/site-img.log"
echo "Для остановки: kill -9 \$(lsof -t -i:7070)"
echo ""
echo "========================================="
echo "  Развертывание завершено успешно!"
echo "========================================="
