# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install spacy
ADD ner_model /code/ner_model
ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]
