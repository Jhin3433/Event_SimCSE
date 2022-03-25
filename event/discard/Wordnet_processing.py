from nltk.corpus import wordnet as wn
import itertools

def get_wordnet_dataset():
    word_info1 = []
    word_info2 = []
    word_info3 = []
    sentence1 = []
    sentence2 = []
    sentence3 = []

    #第一种生成正样本策略
    for word in wn.all_lemma_names():
        for synset in wn.synsets(word):
            if word != synset.lemma_names()[0]: #sense 以当前word开头
                continue
            if(len(synset.examples())) > 2:
                global no_repetitions
                no_repetitions = []
                pos_sense_examples = list(itertools.combinations(synset.examples(), r = 2))
                sentence1.append(pos_sense_examples[0])
                sentence2.append(pos_sense_examples[1])
                word_info1.append(synset._name)
                word_info2.append(synset._name)
                

                print(1)



def test(word):
    syns = wn.synsets(word)
    print(syns[0].name())
    print(syns[0].lemmas()[0].name())

if __name__ == "__main__":
    get_wordnet_dataset()
    # test("program")