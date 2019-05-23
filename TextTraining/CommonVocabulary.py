from Data.TextData import DataManager

def _getDistinctWords(sentences):
    distinct = {}
    counter = 0
    for s in sentences:
        for w in s:
            w = w.lower()
            if (w in distinct):
                distinct[w] += 1
            else:
                distinct[w] = 1
                counter += 1

    return distinct, counter


def generateCommonVocabulary(sentences1, sentences2, outputPath):
    dist1, amount = _getDistinctWords(sentences1)
    print("Corpus 1, unique words:", amount)
    dist2, amount = _getDistinctWords(sentences2)
    print("Corpus 2, unique words:", amount)

    commonWords, commonEcounters = 0, 0
    with open(outputPath, 'w') as file:
        for w in dist1.keys():
            if (w in dist2):
                file.write(w + " - " + str((dist1[w], dist2[w])) + "\n")
                commonEcounters += dist1[w] + dist2[w]
                commonWords += 1

    print("Common number of words:", commonWords)
    print("Number of total encounters:", commonEcounters)
    print("Done!")



if(__name__ == '__main__'):
    audioSentences = DataManager.getFullAudiodescribedSentences()
    subtitleSentences = DataManager.getFullSubtitleSentences()

    generateCommonVocabulary(audioSentences, subtitleSentences, "CommonVocab.txt")