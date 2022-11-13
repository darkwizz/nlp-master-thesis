
POLEVAL_QUESTION_TYPES = ['Gap filling', 'Yes/no', 'Multiple choice', 'Living named entity', 'Named entity', 'Numeric', 'Proper noun', 'Rest']

POLEVAL_QUESTION_PROMPTS = {
    POLEVAL_QUESTION_TYPES[0]: ['Proszę uzupełnić lukę: |&|'],
    POLEVAL_QUESTION_TYPES[1]: ['Proszę podać odpowiedź "tak" czy "nie": |&|'],
    POLEVAL_QUESTION_TYPES[2]: ['Proszę wybrać poprawną opcję z kilku podanych: |&|', 'Zastanawiam się nad tym, którą odpowiedź z podanych jest poprawną. Proszę pomóc z pytaniem: |&|'],
    POLEVAL_QUESTION_TYPES[3]: ['Dane pytanie jest o żywym podmiocie. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[4]: ['To pytanie jest po prostu o podmiocie nazwanym. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[5]: ['Prosze podać liczbę: |&|', 'W tym pytaniu odpowiedź zawiera liczbę. Proszę powiedzieć, |&|'],
    POLEVAL_QUESTION_TYPES[6]: ['W odpowiedzi na dane pytanie jest nazwa własna. Proszę powiedzieć: |&|'],
    POLEVAL_QUESTION_TYPES[7]: ['Proszę odpowiedzieć na pytanie niżej: |&|', 'Zawsze zastanawiałem się: |&|', 'Jaka jest odpowiedź na następujące pytanie: |&|', 'Jest pytanie: |&|']
}
