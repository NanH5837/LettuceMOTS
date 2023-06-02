with open("/home/xiaohu/test/LettuceMOTS/data/lettuceseg/train.txt","a") as f:
    i = 0
    for i in range(0, 51):
        # if i % 7 == 0:
        s = str(i)
        s = s.zfill(6)
        f.write("/home/xiaohu/test/LettuceMOTS/data/lettuceseg/images/0005/{}.png\n".format(s))

