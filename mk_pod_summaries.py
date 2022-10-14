"""Re-run pod summaries over current state of repository

Usage:
  run.py --lang=<language_code>

  run.py (-h | --help)
  run.py --version

Options:
  --lang=<language code>         The language of the Wikipedia to process.
  -h --help                      Show this screen.
  --version                      Show version.

"""

from docopt import docopt
from glob import glob
import joblib
import numpy as np
from collections import Counter


def mk_pod_keywords(page_titles):
    print("POD",page_titles)
    words = [w for title in page_titles for w in title.split()]
    dist = Counter(words)
    return [pair[0] for pair in dist.most_common(50) if pair[0].isalpha()]

def clean_pod_keywords(pod_keywords):
    vocab = []
    for pod in pod_keywords: # pod_keywords is a list of list, length = n of pods
        vocab.extend(pod)
    vocab = list(set(vocab)) # unique words

    clean_keywords = []
    m = np.zeros((len(pod_keywords),len(vocab)))
    for i in range(m.shape[0]):
        for words in pod_keywords[i]:
            m[i][vocab.index(words)] = 1
    col_sum = np.sum(m, axis=0)
    for row in m:
        kw = np.argsort(row / col_sum)[-10:]
        clean_keywords.append([vocab[i] for i in kw])
    return clean_keywords


if __name__ == '__main__':
    args = docopt(__doc__, version='Make pod summaries for a given language, ver 0.1')
    lang = args['--lang']

    pod_list = glob(lang+'/*[0-9].fh')
    pod_summary_mat = np.zeros((len(pod_list),256)) #TODO: Hard-coded hash size. Change?
    pod_keywords = []

    for i in range(len(pod_list)):
        m, page_titles = joblib.load(pod_list[i])
        m = m.toarray()
        #pod_keywords.append(mk_pod_keywords(page_titles))
        keywords = ', '.join(page_titles[:10])+'...'
        pod_keywords.append(keywords)
        print(i,keywords)
        s = np.sum(m, axis=0)
        ns = s / np.linalg.norm(s)
        pod_summary_mat[i] = ns

    #pod_keywords = clean_pod_keywords(pod_keywords)
    pod_list = [p.replace(lang+'/','') for p in pod_list]
    print(pod_summary_mat.shape, len(pod_list), len(pod_keywords))
    print(pod_keywords)

    summary_file = lang+'/'+lang+'wiki.summary.fh'
    joblib.dump([pod_list, pod_keywords, pod_summary_mat],summary_file)
