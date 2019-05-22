from matplotlib import pyplot
from sklearn.decomposition import PCA
import Models.Text.Models as TextModels
import Data.TextData.DataManager as TextManager
from gensim.models import Word2Vec
from Utils import PrintStopwatch


def trainWord2VecFromScratch(corpusPath):
    model = Word2Vec(corpus_file=corpusPath)
    words = list(model.wv.vocab)
    print(words)
    model.save('model.bin')
    # model.wv.save_word2vec_format('model.bin')


def usePretrainedModel(PreTrainedModelPath, outputModelPath, sentences):
    from gensim.models import KeyedVectors

    model_2 = Word2Vec(size=300, min_count=1)
    model_2.build_vocab(sentences)
    total_examples = model_2.corpus_count

    watch = PrintStopwatch("Loading pre-trained model...")
    model = KeyedVectors.load_word2vec_format(PreTrainedModelPath, binary=True)
    watch.printLapTime("Loading finished: ")

    watch.printStartMeasure("Merging Models...")
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format(PreTrainedModelPath, binary=True, lockf=1.0)
    watch.printLapTime("Merge Finished!")

    watch.printStartMeasure("Training Model...")
    model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)
    watch.printLapTime("Training Finished!")

    watch.printStartMeasure("Saving Model...")
    with open(outputModelPath, 'w') as file:
        model.save(file)
    watch.printLapTime("Save Finished!")


def analyzeWord2VecModel(model):
    words = list(model.wv.vocab)
    print(words)
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(result)
    print(model.most_similar(['man']))


def scatterPlotFromVocabulary(model, words):
    pca = PCA(n_components=2)
    result = pca.fit_transform(model[model.wv.vocab])
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def main():
    sentences = TextManager.getFullSubtitleSentences()
    preTrainedModelPath = TextModels.getGooglePreTrainedPath()
    outputPath = "SubtitleModelPreTrained.bin"
    usePretrainedModel(preTrainedModelPath, outputPath, sentences)


if (__name__ == '__main__'):
    main()
