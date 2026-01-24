# 监控脚本：monitor_mem.sh
PID_NAME="python" # 或者是你运行的脚本名
LOG_FILE="memory_trace.log"

echo "开始监控进程: $PID_NAME" > $LOG_FILE
while true; do
    # 获取 Python 进程的总内存占用 (RSS)
    MEM=$(ps -u root -o rss,command | grep "$PID_NAME" | grep -v grep | awk '{sum+=$1} END {print sum/1024}')
    if [ -z "$MEM" ]; then
        echo "$(date): 进程已消失 (可能已被 Kill)" >> $LOG_FILE
        sleep 1
        continue
    fi
    echo "$(date): ${MEM} MB" >> $LOG_FILE
    sleep 0.1
done