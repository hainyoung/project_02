from freezegun import freeze_time
import datetime, time

print(datetime.datetime.now())

# print('import complete')

# @freeze_time("2020-11-16")
# def test():
#     return datetime.datetime.now()

# print(test())


# freezer = freeze_time("2020-11-16 16:45:14")
# freezer.start()
# print(datetime.datetime.now())
# freezer.stop()


# @freeze_time("2020-11-16 16:45:14")
# def test():
#     starttime = datetime.datetime.now()

#     while True:
#         # clock = time.strftime("%Y-%m-%d %H:%M:%S %p", localtime)
#         print(starttime)
#         starttime += datetime.timedelta(seconds=1.0)
#         time.sleep(1)

# print(test())

# def test(timeflow):

# def test(frozen_time):
#     print(datetime.datetime.now())
#     frozen_time.move_to("2020-11-17 17:00:00")
#     print(datetime.datetime.now())

# test()