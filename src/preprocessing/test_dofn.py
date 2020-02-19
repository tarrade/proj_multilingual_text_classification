import pytest, bs4, spacy, re, unidecode

class NLP():
        
    def __init__(self) -> None:
        self.__spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.__spacy.vocab['-pron-'].is_stop = True
        self.__tags = ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios', 'mysql', 'css', 'sql', 'asp.net', 'objective-c', 'ruby-on-rails', '.net', 'c', 'iphone', 'arrays', 'angularjs', 'sql-server', 'ruby', 'json', 'ajax', 'regex', 'r', 'xml', 'asp.net-mvc', 'node.js', 'linux', 'django', 'wpf', 'database', 'xcode', 'vb.net', 'eclipse', 'string', 'swift', 'windows', 'excel', 'wordpress', 'html5', 'spring', 'multithreading', 'facebook', 'image', 'forms', 'git', 'oracle']
        
    def decode_html(self, input_str: str) -> str:
        try:
            soup = bs4.BeautifulSoup(input_str, 'html.parser')
            text_soup = soup.find_all('p')
            code_soup = soup.find_all('code')
        
            text = [val.get_text(separator=' ') for val in text_soup]
            code = [val.get_text(separator=' ') for val in code_soup]
        
            return ' '.join(text), ' '.join(code)
        except TypeError:
            return '', ''
    
    def remove_accented_char(self, input_str: str) -> str:
        try:
            return unidecode.unidecode(input_str)
        except AttributeError:
            return ''
    
    def spacy_cleaning(self, token: str) -> str:
        return not (token.is_stop or token.like_email or token.like_url or token.like_num or token.is_punct)

    def lemmatization(self, input_str: str) -> str:
        doc = self.__spacy(input_str)
        tokens = [token.lemma_.lower() for token in doc if self.spacy_cleaning(token)]
        return ' '.join(tokens)
    
    def remove_punctuation(self, input_str: str) -> str:
        str_punctuation = re.sub('''[!"#$%&\\\\'()*+,-./:;<=>?@[\\]^_`{|}~]+''', ' ', input_str)
        str_white_space = re.sub('''\\s+''', ' ', str_punctuation)
        tokens = str_white_space.split(' ')
        output_list = [token for token in tokens if len(token) > 1]
        return ' '.join(output_list)
    
    def nlp(self, input_str: str) -> str:
        accent = self.remove_accented_char(input_str)
        lemma = self.lemmatization(accent)
        punctuation = self.remove_punctuation(lemma)
        
        return punctuation  
    
    def get_tags(self, tags: list) ->list:
        return list(set(self.__tags) & set(tags))

    def process(self, element):
        title = self.nlp(element['title'])
        text, code = self.decode_html(element['body'])
        text = self.nlp(text)
        code = self.nlp(code)
        tags = self.get_tags(element['tags'])
        
        record = {'id': element['id'],
                 'title': title,
                 'text_body': text,
                 'code_body': code,
                 'tags': tags}
        
        return [record]

@pytest.fixture(scope='module')
def dofn():
    nlp_obj = NLP()
    return nlp_obj
    
@pytest.mark.parametrize('input_str, clean_string_1, clean_string_2',
                         [
                             ('''<p>As you are using this VHOST on port 8080. You need to mention in URL as below.</p><pre><code>http://pizza.local:8080/ </code></pre>''', 'As you are using this VHOST on port 8080. You need to mention in URL as below.', 'http://pizza.local:8080/ '),
                             ('''&lt;p>As you are using this VHOST on port 8080. You need to mention in URL as below.</p> <pre><code>http://pizza.local:8080/ &lt;/code></pre>''', 'As you are using this VHOST on port 8080. You need to mention in URL as below.', 'http://pizza.local:8080/ '),
                             ('', '', '')
                         ]
                         )
def test_decode_html(dofn, input_str, clean_string_1, clean_string_2):
    assert dofn.decode_html(input_str) == (clean_string_1, clean_string_2)
    
@pytest.mark.parametrize('raw_string, cut_string',
                         [
                             ('Jeder Mensch sollte eine Ausbildung geniessen dürfen. Was meinst du, René?', 'Jeder Mensch sollte eine Ausbildung geniessen durfen. Was meinst du, Rene?'),
                             ('Jeder Mensch sollte eine Ausbildung geniessen d\\xc3\\xbcrfen. Was meinst du, Ren\\xc3\\xa9.', 'Jeder Mensch sollte eine Ausbildung geniessen d\\xc3\\xbcrfen. Was meinst du, Ren\\xc3\\xa9.'),
                             ('', '')
                         ]
                         )
def test_remove_acctented_char(dofn, raw_string, cut_string):
    assert dofn.remove_accented_char(raw_string) == cut_string
    
@pytest.mark.parametrize('raw_string, clean_string',
                         [
                             ('Swift: Drag a UIView between two other UIViews', 'swift drag uiview uiview'),
                             ('', '')
                         ]
                         )
def test_lemmatization(dofn, raw_string, clean_string):
    assert dofn.lemmatization(raw_string) == clean_string

@pytest.mark.parametrize('raw_string, clean_string',
                         [
                             (''''
    def calculator(x:str, y:list) ->int:
        multiplication=[x*element for element in y]
        return sum(multiplication)''', 'def calculator str list int multiplication element for element in return sum multiplication'),
                             ('', '')
                         ]
                         )
def test_remove_punctuation(dofn, raw_string, clean_string):
    assert dofn.remove_punctuation(raw_string) == clean_string