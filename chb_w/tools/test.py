import nltk

def res(text):
    w = nltk.word_tokenize(text)
    for word in w:
        if word.lower() == "hello" or word.lower() == "hi" or word.lower() == "hola":
            return "> Hi"
        if word.lower() == "":
            return "> I didn't understand you"
        if word.lower() == "depressed" or word.lower() == "unhappy":
            return "> Everyone sees darks day, they just pass by"
        if word.lower() == "help":
            return "> you can use our website to manage your daily chores and your money"


if __name__ == "__main__":
    s = input()
    while(s != ""):
        print(res(s))
        s = input()