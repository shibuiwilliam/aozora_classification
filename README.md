# aozora_classification
This project aims to predict how well a Japanese sentence is similar to some Japanese classical author's ones.
The Japanese authors include Soseki Natsume(夏目漱石), Ogai Mori(森鴎外), Ryunosuke Akutagawa(芥川龍之介) and so on.


## Getting data from Aozora-bunko(青空文庫)
I'm getting training data from Aozora-bunko, which is a public repository for Japanese classical pieces with outdated copyright.
http://www.aozora.gr.jp/

In order to download text pieces from Aozora-bunko, I made aozora_scrape.py.
It downloads all the pieces in zip file of certain authors, specified in target_author.csv, extracts the zips to SHIFT-JIS encoded text files, converts them to utf-8 and finally changes the file format to csv.
The last csv has each lines splitted by "。", the Japanese form of period, and new line feed.


## Training the model
Thanks to the following paper and blog, I use character-level convolurional neural network to train the classification model for Japanese sentences.
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
http://qiita.com/bokeneko/items/c0f0ce60a998304400c8

After downloading and changing the text to csv, run the aozora_cnn.py to generate classification model.


## Classification
As you have finished generating the model, now you are ready to use it.
