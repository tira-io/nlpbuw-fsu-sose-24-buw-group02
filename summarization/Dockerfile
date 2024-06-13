FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install transformers google-api-python-client google-auth nltk pandas
ADD run.py /code/run.py
ADD credentials.json /code/credentials.json

ENTRYPOINT [ "python3", "/code/run.py" ]
