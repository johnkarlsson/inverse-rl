ps aux | grep bin/main.o | grep -v grep | awk '{print $2}' | xargs sudo kill
