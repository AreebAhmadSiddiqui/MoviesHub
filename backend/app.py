from flask import Flask, request, jsonify
import traceback
import pandas as pd
import pickle
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

def stemmer(tag):
    ps = PorterStemmer()
    list = []
    for x in tag.split():
        list.append(ps.stem(x))
    return " ".join(list)

def getCrewDetails(type,data):
    if(type=='cast'):
        return [x['name'] for x in data]
    else:
        list=[]
        for x in data:
            if x['job']=='Director' or x['job']=='Editor' or x['job']=='Producer':
                list.append(x['name'])
        return list

def getMovieData(movie_id):

        response_API = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8f655507141a5d524fc2024c9f76b6c7&language=en-US").json()

        genres=[x['name'] for x in response_API['genres']]
        genres=[x.replace(' ','') for x in genres]

        getKeyWords=requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key=8f655507141a5d524fc2024c9f76b6c7').json()
        keywords=[x['name'] for x in getKeyWords['keywords']]
        keywords=[x.replace(' ','') for x in keywords]


        id=response_API['id']
        title=response_API['title']
        overview=response_API['overview'].split()
        tagline=response_API['tagline'].split()

        movie_credits=requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key=8f655507141a5d524fc2024c9f76b6c7&language=en-US').json()
        cast=getCrewDetails('cast',movie_credits['cast'])

        cast=[x.replace(' ','') for x in cast]

        movie_tag=genres+keywords+overview+cast+tagline
        movie_tag=stemmer(" ".join(movie_tag).lower())

        data={'id':[id],
            'title':[title],
            'tags':[movie_tag]
            }

        return data
        
app = Flask(__name__)

CORS(app)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json()

        data=getMovieData(data['id'])
        new_movies = pickle.load(open('new_movies.pkl','rb'))
        new_movies=pd.DataFrame(new_movies)
        movie_details = pd.DataFrame(data)

        # Create a Vectorizer Object
        vectorizer = pickle.load(open('vectorizer.pkl','rb'))

        # Encode the document
        vectors = pickle.load(open('vectors.pkl','rb'))

        x=vectorizer.transform(movie_details['tags'])
        s=cosine_similarity(x,vectors)
        moLi=s[0].tolist()
        movies_list=sorted(enumerate(moLi),reverse=True,key=lambda x:x[1])[1:20]
        movies_id=[]
        for i in movies_list:
            movies_id.append(new_movies.iloc[i[0]].id)
        return jsonify({'prediction': str(movies_id)})

    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5000,debug=True)
