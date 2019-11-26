from keras.models import Sequential
from keras.models import load_model
from keras import layers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import sqrt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constantes que usaremos em nosso código
MAX_SEQUENCE_LENGTH = 5000
MAX_NUM_WORDS = 25000
TEXT_DATA = '/home/diego/Documents/unb/pw/Fakenews-Detection/data/fake_or_real_news.csv'
texto = "Your daily reality snack Georgia Abandons Ukraine's Anti-Russian Obsession After a brief period in which both Ukraine and Georgia appeared to be united against Russia, it now appears that the two nations are moving along very different paths Originally appeared at Russia Direct In October, Georgia didn’t support any of Ukraine’s resolutions denouncing the Kremlin’s foreign policy within the Parliamentary Assembly of the Council of Europe (PACE). That is surprising, given how many analysts had by now assumed that Georgia and Ukraine were on the same page when it came to Russia. The two resolutions deal with “the political implications of Russia’s aggression in Ukraine” and human rights abuses “on the occupied Ukrainian territories.” By supporting them, PACE recognized the military conflict in Ukraine as “Russian aggression” for the first time and called on the Kremlin to withdraw its forces from the eastern part of Ukraine. Moreover, it denounced the parliamentary elections , recently conducted by Russia in Crimea . When the Georgian delegation in PACE didn’t support these resolutions, the nation’s pro-Western parties reacted strongly. For example, the United National Movement lambasted the Georgian government and accused the country’s former Prime Minister, Bidzina Ivanishvili, of supporting Russia. Moreover, Mikheil Saakashvili, the former Georgian president and now the governor of the Odessa region in Ukraine, described such a stance as a “disgraceful” move. However, an immediate response came from one of the members of the Georgian delegation in PACE, Eka Beselia. She retorted that Tbilisi needed to defend its own national interests. Even though this statement seems to have alleviated the increasing conflict, the video of Russian-Ukrainian journalist Matwey Ganapolsky, who accuses Georgia of betraying Ukraine in favor of Russia, fuelled the tensions. In contrast, Russian pundits see the unwillingness of Georgia to vote for the PACE resolution as a sign of improvement in Tbilisi-Moscow relations. In reality, the reluctance of the Georgian Dream, the ruling party in Georgia, to approve these resolutions is just the logical conclusion of complicated relations with Kiev. Since the start of the color revolutions in the post-Soviet space , Georgia and Ukraine were largely in the same boat. After the success of the Rose Revolution in Tbilisi and the Orange Revolution in Kiev, the newly elected governments were closely connected with each other and teamed up against Russia. This resulted from friendly relations between Ukraine’s former prime minister Yulia Timoshenko and former president Viktor Yushchenko on the one hand, and Georgia’s Saakashvili on the other hand. However, their relationship was rather pragmatic in its nature, although officially Tbilisi recognized Ukraine as one of its closest allies. Since 2007 the democratic processes in the two countries have started moving in a reverse direction. Saakashvili’s penchant for conducting an aggressive policy as well as his authoritarian inclinations was increasing, while Ukraine faced the corruption and the political rivalry between Timoshenko and Yushchenko. The more impact this had on the countries’ stability and development, the more obvious became the fact that the ruling elites from both sides did not support democratic reforms, but only the regimes that were friendly to them. Thus, Georgian-Ukrainian relations could be seen as a form of cooperation between governments, not between the people. And this trend became relevant until the 2010 presidential elections, when Georgia’s civil society and population called on the government to support democratic processes and regime change in its “brother” country. From then on, Georgia has been shying away from supporting the political regime in Ukraine and focusing more on the support of the country’s own population. However, Ukraine refused to consider such tactics, with its official representatives criticizing the Georgian Dream coalition for supporting Russia during the 2012 parliamentary elections. Moreover, Kiev cooperated with Georgia’s United National Movement, which was openly accused of building an authoritarian regime and egregious human rights abuses. Logically, the new Georgian government under Ivanishvili cannot help paying attention to this fact. But it was relatively reticent and didn’t respond, even when Georgian volunteers came to fight in Eastern Ukraine to support Kiev and accused Tbilisi of supporting Russia. That had some implications for the Georgian Dream: It was seen as a political force that is capable of defending the country’s national interests. Moreover, Georgian voters also saw the fact that Saakashvili was appointed as the governor of the Odessa region as an unfriendly move from Ukraine, as a slap in the face, because the former Georgian president was legally prosecuted in his home country, which meant that Ivanishvili couldn’t fulfill his pledges and restore justice [During the election campaign he promised to put Saakashvili in jail for corruption and the abuse of power — Editor’s note]. The problem was exacerbated when Kiev granted Saakashvili Ukrainian citizenship, which made it impossible to imprison the former Georgian president. Saakashvili crossed the red line during the latest parliamentary elections in Georgia during the campaign. First, his colleagues from the United National Movement visited Ukraine. Second, he openly called for a coup d’état against the Georgian government, which he sees as pro-Russian. In fact, he threatened to conduct a new revolution in Georgia. This was the last straw for the Georgian Dream. It is safe to say that the current Georgian political elites started seeing Ukraine as a real headache and the shelter for dubious and controversial Georgian politicians from the United National Movement accused of different wrongdoings and legal violations. However, with the victory of the Georgian Dream in the 2016 parliamentary elections, a lot has changed. Moreover, the odds of the party of winning the constitutional majority are really high. It means that the influence of the party is growing in the Georgian parliament and even more could change. As a result, the government won’t necessarily have to take into account the views of other political forces to take decisions. It can be pretty outspoken now that it won’t put up with anti-government moves and initiatives like the ones promoted by Saakashvili. Moreover, the Georgian voters, who are seeking to have those involved in the violations during Saakashvili’s tenure prosecuted. So, in this regard, the electorate supports the Georgian Dream. Thus, all this indicates that Georgian-Ukrainian relations have always been more complex and nuanced than they seemed to be at first glance. During Saakashvili’s tenure, there was cooperation between his government and the ones of Timoshenko and Yushchenko. However, eventually, Tbilisi shifted its priority from supporting top political officials to supporting society and people. Ukrainian politicians should keep in mind that the Russian factor is not the only one that determines the Ukrainian-Georgian agenda. Providing shelter to Saakashvili also does matter. So, to improve the relations with Tbilisi, Kiev should take into account its national interest and support the Georgian people instead of the country’s politicians."



# Leitura da base de dados e Limpeza dos dados (remover colunas de texto em branco, etc...)
df = pd.read_csv(TEXT_DATA)
df.drop(labels=['id', 'title'], axis='columns', inplace=True)
mask = list(df['text'].apply(lambda x: len(x) > 0))
df = df[mask]


# Como utilizamos aprendiagem supervisionada separamos em dois dataframes um com os textos e outro com as labels (FAKE/REAL)
texts = df['text']
labels = df['label']


# Criação dos tokens que iremos enviar para a rede convolucional
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
token_text = tokenizer.texts_to_sequences([texto])[0]
text_padding = pad_sequences([token_text],
                             maxlen=MAX_SEQUENCE_LENGTH,
                             padding='pre',
                             truncating='pre')

# Leitura do modelo que fo salvo pelo outro arquivo
new_model = load_model('./model.h5')

# Mostrar informações da Rede
new_model.summary()


print(1 - new_model.predict(text_padding)[0][0])
