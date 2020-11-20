# import time

# while True:
#     localtime = time.localtime()
#     clock = time.strftime("%Y-%m-%d %I:%M:%S %p", localtime)
#     print(clock)
#     time.sleep(1)


import datetime, time

# now = datetime.now()
# # datetime = now.strftime('%Y-%m-%d %H:%M:%S')
# # datetime = now.strftime('%H:%M:%S')
# # print(datetime)

t = datetime.datetime(2020, 11, 16, 16, 45, 58)

while True:
    # t = time(16, 45, 14)
    # t = t.isoformat()
    print(t)
    t += datetime.timedelta(seconds=1.0)
    time.sleep(1)


# starttime = datetime(2020, 11, 16, 16, 45, 14)
# print(now-starttime)