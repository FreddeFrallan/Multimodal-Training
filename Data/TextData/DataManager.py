import RootDir

FOLDER_DIR = "Data/TextData/"
FULL_SUBTITLE_FILE_NAME = "FullSubtitle.txt"
FULL_AUDIODESCRIBED_FILE_NAME = "FullAudiodescribed.txt"


def _getAbsoluteFolderPath(fileName):
    return RootDir.getAbsolutePath(FOLDER_DIR + fileName)


def getFullSubtitleCorpusPath():
    return _getAbsoluteFolderPath(FULL_SUBTITLE_FILE_NAME)


def getFullAudiodescribedCorpusPath():
    return _getAbsoluteFolderPath(FULL_AUDIODESCRIBED_FILE_NAME)


def getFullSubtitleCorpus():
    with open(_getAbsoluteFolderPath(FULL_SUBTITLE_FILE_NAME)) as file:
        return file.readlines()


def getFullAudiodescribedCorpus():
    with open(_getAbsoluteFolderPath(FULL_AUDIODESCRIBED_FILE_NAME)) as file:
        return file.readlines()

def _splitLinesIntoSentences(lines):
    return [l.strip().split(' ') for l in lines]

def getFullSubtitleSentences():
    return _splitLinesIntoSentences(getFullSubtitleCorpus())


def getFullAudiodescribedSentences():
    return _splitLinesIntoSentences(getFullAudiodescribedCorpus())
