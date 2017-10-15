

def readfile(fileName):
    data = ""
    try:
        f = open(fileName, "r")
        data = f.readlines()
        f.close()
    except Exception as e:
        print (e)
        exit(-1)
    return data



def main():
    data = readfile("../data/train")
    speakers = []
    for line in data:
        line = line.strip()
        speakerName = line.split(' ')[0]
        speakers.append(speakerName)
    for i in set(speakers):
        print (i)


if __name__=="__main__":
    print("Hello")
    main()

