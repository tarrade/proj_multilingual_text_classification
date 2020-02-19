import os, sys, pathlib, re, string, unidecode, spacy, bs4, apache_beam as beam, time

class NLP(beam.DoFn):
        
    def setup(self) -> None:
        self.__spacy = spacy.load('en_core_web_sm')
        self.__spacy.vocab['-pron-'].is_stop = True
        self.__tags = ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios', 'mysql', 'css', 'sql', 'asp.net', 'objective-c', 'ruby-on-rails', '.net', 'c', 'iphone', 'arrays', 'angularjs', 'sql-server', 'ruby', 'json', 'ajax', 'regex', 'r', 'xml', 'asp.net-mvc', 'node.js', 'linux', 'django', 'wpf', 'database', 'xcode', 'vb.net', 'eclipse', 'string', 'swift', 'windows', 'excel', 'wordpress', 'html5', 'spring', 'multithreading', 'facebook', 'image', 'forms', 'git', 'oracle']
        
    def __debug(self, id_element: int, body_str: str, code_str: str) -> None:
        body_length = len(body_str.split(' '))
        code_length = len(code_str.split(' '))
        print_statement = '- DEBUG_DF - id_df:{0}, body_df:{1}, code_df:{2}'.format(id_element, body_length, code_length)
        
        print(print_statement)
        
    def __decode_html(self, input_str: str) -> str:
        try:
            soup = bs4.BeautifulSoup(input_str, 'html.parser')
            text_soup = soup.find_all('p')
            code_soup = soup.find_all('code')
        
            text = [val.get_text(separator=' ') for val in text_soup]
            code = [val.get_text(separator=' ') for val in code_soup]
        
            return ' '.join(text), ' '.join(code)
        except TypeError:
            return '', ''
    
    def __remove_accented_char(self, input_str: str) -> str:
        try:
            return unidecode.unidecode(input_str)
        except AttributeError:
            return ''
    
    def __spacy_cleaning(self, token: str) -> str:
        return not (token.is_stop or token.like_email or token.like_url or token.like_num or token.is_punct)

    def __lemmatization(self, input_str: str) -> str:
        doc = self.__spacy(input_str)
        tokens = [token.lemma_.lower() for token in doc if self.__spacy_cleaning(token)]
        return ' '.join(tokens)
    
    def __remove_punctuation(self, input_str: str) -> str:
        str_punctuation = re.sub('''[!"#$%&\\\\'()*+,-./:;<=>?@[\\]^_`{|}~]+''', ' ', input_str)
        str_white_space = re.sub('''\s+''', ' ', str_punctuation)
        tokens = str_white_space.split(' ')
        output_list = [token for token in tokens if len(token) > 1 and not token.isdigit()]
        return ' '.join(output_list)
    
    def __nlp(self, input_str: str) -> str:
        accent = self.__remove_accented_char(input_str)
        lemma = self.__lemmatization(accent)
        punctuation = self.__remove_punctuation(lemma)
        
        return punctuation  
    
    def __get_tags(self, tags: list) ->list:
        return list(set(self.__tags) & set(tags))

    def process(self, element):
        text, code = self.__decode_html(element['body'])
        self.__debug(element['id'], text, code)
        title = self.__nlp(element['title'])
        
        start_time = time.time()
        
        text = self.__nlp(text)
        code = self.__nlp(code)
        
        end_time = time.time()

        total_time = end_time-start_time
        print("- DEBUG_DF_B - id_df:{0}, time_df:{1}".format(element['id'], total_time))
        
        if total_time >= 300:
            print("TIME-ERROR2 - id_df:{0}".format(element['id']))
        elif total_time >=60:
            print("TIME-ERROR1 - id_df:{0}".format(element['id']))
        elif total_time < 60:
            print("TIME-OK - id_df:{0}".format(element['id']))
        
        tags = self.__get_tags(element['tags'])
        
        record = {'id': element['id'],
                 'title': title,
                 'text_body': text,
                 'code_body': code,
                 'tags': tags}
        
        return [record]
    
class CSV(beam.DoFn):
    def process(self, element):
        text_line = str(element['id']) + ', ' + str(element['title'])  + ', ' + str(element['text_body']) \
                     + ', ' + str(element['code_body']) + ', ' + '-'.join(element['tags'])
        return [text_line]