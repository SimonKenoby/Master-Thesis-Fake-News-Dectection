from dataset1 import dataset1

if __name__ == "__main__":
    ds = dataset1('liar_liar', {"split" : "train"}, 10, "Dictionary.dct")
    for i in range(0, len(ds)):
        print(ds[i])
        break
