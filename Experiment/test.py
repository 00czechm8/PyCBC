import time

duration = 10
fs = 1e3

for idx in range(int(duration*fs)):
    print("Time: ", idx/fs)
    time.sleep(1/fs)