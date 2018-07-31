from __future__ import print_function

from plot_utils import plot_and_save_history
from model import Seq2SeqSummarizer
from hps import hps

LOAD_EXISTING_WEIGHTS = True

report_dir_path = './reports'
model_dir_path = './models'

summarizer = Seq2SeqSummarizer(hps)
summarizer.fit(load_weights=False, epochs=20)

# history = summarizer.fit(load_weights=False, epochs=100)

# history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history-v' + str(summarizer._version) + '.png'

# plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})