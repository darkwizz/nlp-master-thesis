

MKQA_QUESTION_TYPES = ['binary', 'date', 'entity', 'number', 'number_with_unit', 'short_phrase']
MKQA_QUESTION_PROMPTS = {
    MKQA_QUESTION_TYPES[0]: ['Proszę podać odpowiedź "tak" czy "nie": |&|', 'Dane pytanie jest binarne, proszę odpowiedzieć: |&|'],
    MKQA_QUESTION_TYPES[1]: ['Proszę podać datę w odpowiedzi: |&|'],
    MKQA_QUESTION_TYPES[2]: ['Dane pytanie jest o podmiocie. Proszę powiedzieć: |&|'],
    MKQA_QUESTION_TYPES[3]: ['Prosze podać liczbę: |&|', 'W tym pytaniu odpowiedź zawiera liczbę. Proszę powiedzieć, |&|'],
    MKQA_QUESTION_TYPES[4]: ['Proszę podać liczbę i jednostkę w odpowiedzi na pytanie: |&|'],
    MKQA_QUESTION_TYPES[5]: ['Odpowiedzią na pytnie poniżej powinno być krótkie zdanie. Proszę powiedzieć: |&|']
}