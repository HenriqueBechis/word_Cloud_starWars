from nltk import tokenize
import pandas as pd
from LeIA.leia import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

s = SentimentIntensityAnalyzer()

SW_episodeIV = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 
SW_episodeVI = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 
SW_episodeV = pd.read_csv("SW_EpisodeIV_ptBR - SW_EpisodeIV_ptBR.csv", encoding='utf-8') 

roteiro = pd.concat([SW_episodeIV, SW_episodeVI, SW_episodeV])
roteiro = roteiro.iloc[:, :3]

freq_pos_personagem = {}
freq_neg_personagem  = {}
freq_neu_personagem  = {}

for i, dialog in enumerate(roteiro.iloc[:, 2:3].values):
    frase = dialog[0]
    nome = roteiro.iloc[i, 1:2].values[0]
    print(nome)

    if nome not in freq_pos_personagem:
        freq_pos_personagem[nome] = 0
    if nome not in freq_neg_personagem:
        freq_neg_personagem[nome] = 0
    if nome not in freq_neu_personagem:
        freq_neu_personagem[nome] = 0

    
    lines_list = tokenize.sent_tokenize(frase, language='portuguese')
    
    for line in lines_list:
        score = s.polarity_scores(line)
        
        if score['compound'] > 0:
            freq_pos_personagem[nome] += 1
        elif score['compound'] < 0:
            freq_neg_personagem[nome] += 1
        else:
            freq_neu_personagem[nome] += 1



#WordCloud positivo
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(freq_pos_personagem)
plt.figure()
plt.title("Nomes dos personagens com falas positivas")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#WordCloud negativo
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(freq_neg_personagem)
plt.figure()
plt.title("Nomes dos personagens com falas negativas")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#WordCloud neutro
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(freq_neu_personagem)
plt.figure()
plt.title("Nomes dos personagens com falas neutras")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()