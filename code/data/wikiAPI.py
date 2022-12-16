import wikipediaapi
import re
                                                                                                                                                                                                                                                                                                                       
def wiki(class_name):
    wiki = wikipediaapi.Wikipedia(language='ko')
    p_wiki = wiki.page(class_name)
    #페이지 있으면 페이지 출력
    if(p_wiki.exists()):
        return p_wiki.text

def download_txt(text, keyword):
    f = open(f'./{keyword}.txt', 'w')
    f.write(text)
    f.close()

#중국어 제거
def remove_foreign_chinese(text):
    new_text = text
    china = re.compile(u"[\U00004E00-\U00009FFF]",flags=re.UNICODE)
    new_text = china.sub('', new_text)
    new_text = re.sub(' +', ' ', new_text)
    new_text = re.sub('\(\)|\( \)', '', new_text)
    return new_text

def text_processing(text):
    new_txt = text.replace('. ', '.\n')
    return new_txt

keyword = '' #put in your keyword in here
txt = wiki(keyword)
txt = remove_foreign_chinese(txt)
txt = text_processing(txt)
download_txt(txt, keyword)