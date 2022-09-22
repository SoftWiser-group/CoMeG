from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def compute_accuracy(reference_strings, predicted_strings):
    assert(len(reference_strings) == len(predicted_strings))
    correct = 0.0
    for i in range(len(reference_strings)):
        if reference_strings[i] == predicted_strings[i]:
            correct += 1
    return 100 * correct/float(len(reference_strings))


def compute_bleu(references, hypotheses):
    bleu_4_sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_4_sentence_scores.append(sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method2))
    return 100*sum(bleu_4_sentence_scores)/float(len(bleu_4_sentence_scores))


def compute_sentence_bleu(ref, hyp):
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method2)


def compute_sentence_meteor_rouge(reference_list, sentences):
    preds = dict()
    refs = dict()

    for i in range(len(sentences)):
        preds[i] = [' '.join([''.join(s.split()) for s in sentences[i]])]
        refs[i] = [' '.join(l) for l in reference_list[i]]

    final_scores = dict()

    scorers = [
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                final_scores[m] = scs
        else:
            final_scores[method] = scores

    return final_scores['METEOR'], final_scores['ROUGE_L']


def compute_meteor_rouge(reference_list, sentences):
    """
    return: METEOR, ROUGE_L
    """
    meteors, rougels = compute_sentence_meteor_rouge(reference_list, sentences)
    return 100 * sum(meteors)/len(meteors), 100 * sum(rougels)/len(rougels)


def test():
    references = [[['Modified', 'the', 'code', 'to', 'work', 'with', 'Python', '3']],
                  [['Add', 'the', 'following', 'in', 'System', 'Variables']]]
    pred_instances = [['Modified', 'the', 'code', 'to', 'work', 'with', 'Python', '2'],
                      ['Add', 'the', 'following', 'in', 'System', 'Variables']]
    print('Predicted BLEU: {}'.format(compute_bleu(references, pred_instances)))
    meteor, rougel = compute_meteor_rouge(references, pred_instances)
    print('Predicted Meteor: {}\nPredicted ROUGE_L: {}\n'.format(meteor, rougel))


if __name__ == "__main__":
    test()
