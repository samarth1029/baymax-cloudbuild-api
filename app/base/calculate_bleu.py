from nltk.translate.bleu_score import sentence_bleu


def get_bleu(reference, prediction):
    """
    Given a reference and prediction string, outputs the 1-gram,2-gram,3-gram and 4-gram bleu scores
    """
    reference = [reference.split()]  # should be in an array (cos of multiple references can be there here only 1)
    prediction = prediction.split()
    bleu1 = sentence_bleu(reference, prediction, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, prediction, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, prediction, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, prediction, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4
