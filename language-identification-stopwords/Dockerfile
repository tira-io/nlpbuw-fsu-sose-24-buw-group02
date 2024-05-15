FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD stopwords /code/stopwords
ADD run.py /code/run.py
ADD model.joblib /code/model.joblib
ADD char_vectorizer.joblib /code/char_vectorizer.joblib
ADD word_vectorizer.joblib /code/word_vectorizer.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]
