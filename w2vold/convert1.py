from w import *

if __name__ == "__main__":
    #sentences = "aap noot mies wim zus jet aap noot mies aap aap".split(" ")

    vocab, solution = load("trainf.w")
    save("trainb.w", vocab, solution, binary=True)


