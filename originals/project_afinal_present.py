import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
import nltk



plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

df = pd.read_csv('fake_news.csv')




print("Distribution des étiquettes (Vrai/Faux) :")
print(df['Etiquette'].value_counts())
print("\nValeurs manquantes initiales :")
print(df.isnull().sum())


df = df.fillna('')
df.drop_duplicates(inplace=True)


print("\nBoxplot de Nb_Partages (échelle log - AVANT Capping) :")
sns.boxplot(data=df, x='Etiquette', y='Nb_Partages', palette='Set2')
plt.yscale('log') 
plt.title('Distribution des Partages (Avant Capping)')
plt.show()


Q1 = df['Nb_Partages'].quantile(0.25)
Q3 = df['Nb_Partages'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3.0 * IQR 


df['Nb_Partages'] = np.where(
    df['Nb_Partages'] > upper_bound,
    upper_bound,
    df['Nb_Partages']
)

print(f"\nLes outliers de Nb_Partages ont été plafonnés à la valeur: {upper_bound:.0f}")


print("\nBoxplot de Nb_Partages (échelle log - APRÈS Capping) :")
sns.boxplot(data=df, x='Etiquette', y='Nb_Partages', palette='Set2')
plt.yscale('log')
plt.title('Distribution des Partages (Après Capping)')
plt.show()



df['Body_Length'] = df['Corps_Textee'].str.len()
sns.histplot(data=df, x='Body_Length', hue='Etiquette', kde=True, palette='coolwarm')
plt.title('Distribution de la Longueur du Corps de Texte')
plt.show()

df['Source_Domain'] = df['URL_Source'].apply(lambda x: x.split('/')[2] if isinstance(x, str) and len(x.split('/')) > 2 else 'Unknown')



df['Title_Word_Count'] = df['Title'].apply(lambda x: len(str(x).split()))
df['Title_Char_Count'] = df['Title'].apply(lambda x: len(str(x)))


df['Body_Word_Count'] = df['Corps_Textee'].apply(lambda x: len(str(x).split()))
df['Body_Char_Count'] = df['Corps_Textee'].apply(lambda x: len(str(x)))
df['Body_Avg_Word_Length'] = df['Body_Char_Count'] / (df['Body_Word_Count'] + 1)


def detect_clickbait(text):
    text = str(text).lower()
    score = 0
    if '!' in text: score += 1
    if '?' in text: score += 1
    if 'video' in text: score += 1
    if 'exclusif' in text: score += 1
    if text.isupper(): score += 2 
    return score

df['Clickbait_Score'] = df['Title'].apply(detect_clickbait)


df['Shares_Per_Char'] = df['Nb_Partages'] / (df['Body_Char_Count'] + 1)


df['Log_Shares'] = np.log1p(df['Nb_Partages'])


labels = ['Low', 'Popular', 'Viral']
bins = [-1, 1000, 10000, np.inf] 
df['Virality_Category'] = pd.cut(df['Nb_Partages'], bins=bins, labels=labels)


target_map = {'Fake': 1, 'Real': 0}
df['Target'] = df['Etiquette'].map(target_map)




features = ['Title_Word_Count', 'Title_Char_Count',
            'Body_Word_Count', 'Body_Char_Count', 'Body_Avg_Word_Length',
            'Clickbait_Score', 'Shares_Per_Char', 'Log_Shares']

X = df[features]
y = df['Target']


X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- RÉSULTATS DE LA CLASSIFICATION ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Matrice de Confusion')
plt.show()

importances = model.feature_importances_
feature_names = X.columns
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Importance des Features (Facteurs Clés)')
plt.show()




fake_titles = df[df['Etiquette'] == 'Fake']['Title'].fillna('')

vectorizer = CountVectorizer(stop_words='english', max_features=20)

X_counts = vectorizer.fit_transform(fake_titles)
sum_words = X_counts.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("\n--- BONUS 1: TOP 10 des mots-clés dans les Fake News ---")
for word, freq in words_freq[:10]:
    print(f"- {word}: {freq}")




share_threshold = df['Nb_Partages'].quantile(0.80)
length_threshold = 500

bots = df[(df['Nb_Partages'] > share_threshold) & (df['Body_Char_Count'] < length_threshold)]

print(f"\n--- BONUS 2: Nombre d'articles suspects 'Bot-like' détectés : {len(bots)} ---")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Body_Char_Count', y='Nb_Partages', hue='Etiquette', alpha=0.6)
plt.axhline(y=share_threshold, color='r', linestyle='--', label='Seuil Partages Élevés')
plt.axvline(x=length_threshold, color='r', linestyle=':', label='Seuil Texte Court')
plt.title('Détection de Comportement "Bot-like" (Partages vs Longueur)')
plt.legend()
plt.show()




y_viral = df['Nb_Partages']

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_viral, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_v, y_train_v)

y_pred_v = regressor.predict(X_test_v)
mae = mean_absolute_error(y_test_v, y_pred_v)

print("\n--- BONUS 3: PRÉDICTION DE VIRALITÉ ---")
print(f"Erreur Moyenne Absolue (MAE) : {mae:.0f} partages")

comparison = pd.DataFrame({'Réel': y_test_v, 'Prédit': y_pred_v})
print("\nAperçu des prédictions :")
print(comparison.head())

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = tfidf.fit_transform(df['Corps_Textee'].fillna(''))

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X_text)

df['Cluster_Sujet'] = kmeans.labels_

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

print("\n--- BONUS 4: Mots-clés principaux par Cluster (Sujet) ---")
for i in range(num_clusters):
    print(f"Cluster {i} : ", end='')
    for ind in order_centroids[i, :10]: 
        print(f"{terms[ind]}, ", end='')
    print() 

sns.countplot(x='Cluster_Sujet', hue='Etiquette', data=df)
plt.title('Répartition des Fake News par Sujet (Cluster)')
plt.show()