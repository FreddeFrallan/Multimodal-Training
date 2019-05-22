from gensim import models
from gensim.models import Word2Vec

import RootDir

FOLDER_DIR = "Models/Text/"
SUBTITLE_MODEL = "RawFullsubtitleWord2Vec.bin"
GOOGLE_PRE_TRAINED_MODEL = "GoogleNews-vectors-negative300.bin"
AUDIODESCRIBED_PRE_TRAINED_MODEL = "/AudiodescribedPretrained/SubtitleModelPreTrained.bin"
SUBTITLE_PRE_TRAINED_MODEL = "/SubtitleModel/SubtitleModelPreTrained.bin"


def _getAbsoluteFolderPath(modelName):
    return RootDir.getAbsolutePath(FOLDER_DIR + modelName)


def getRawSubtitleWord2VecModel():
    return  Word2Vec.load(_getAbsoluteFolderPath(SUBTITLE_MODEL))

def getGooglePreTrainedPath():
    return _getAbsoluteFolderPath(GOOGLE_PRE_TRAINED_MODEL)

def getAudiodescribedPretrainedPath():
    return _getAbsoluteFolderPath(AUDIODESCRIBED_PRE_TRAINED_MODEL)

def getGoogleKeyedVectors():
    return models.KeyedVectors.load_word2vec_format(getGooglePreTrainedPath(), binary=True)

def getAudiodescribedPretrainedKeyedVectors():
    return models.KeyedVectors.load(getAudiodescribedPretrainedPath())