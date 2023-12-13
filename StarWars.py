from nltk import tokenize
import pandas as pd
from LeIA.leia import SentimentIntensityAnalyzer 

s = SentimentIntensityAnalyzer()

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

SW_episodeIV = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 
SW_episodeVI = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 
SW_episodeV = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 

roteiro = pd.concat([SW_episodeIV,SW_episodeVI,SW_episodeV])
roteiro = roteiro.iloc[:, :3]

for i, dialog in enumerate(roteiro.iloc[:, 2:3].values):
    frase = dialog[0]
    nome = roteiro.iloc[i, 1:2].values[0]
    print(nome)
    lines_list = tokenize.sent_tokenize(frase, language='portuguese')

    for line in lines_list:
        score = s.polarity_scores(line)
        #print(line, score)




