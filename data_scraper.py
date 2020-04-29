import urllib
import urllib.request
import os
from requests import get
from bs4 import BeautifulSoup
from pydub import AudioSegment
import pandas as pd
import numpy as np

# from http://accent.gmu.edu/browse_language.php, return list of languages
def get_languages():
    url = "http://accent.gmu.edu/browse_language.php"
    html = get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    languages = []
    language_lists = soup.findAll('ul', attrs={'class': 'languagelist'})
    for ul in language_lists:
        for li in ul.findAll('li'):
            languages.append(li.text)
    return languages

# from list of languages, return list of urls
def get_language_urls(lst):
    urls = []
    for language in lst:
        urls.append('http://accent.gmu.edu/browse_language.php?function=find&language={}'.format(language))
    return urls

# from language, get the number of speakers of that language
def get_num(language):
    url = 'http://accent.gmu.edu/browse_language.php?function=find&language=' + language
    html = get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    test = soup.find_all('div', attrs={'class': 'content'}) 
    try:
        num = int(test[0].find('h5').text.split()[2])      
    except AttributeError:
        num = 0
    except IndexError:
        num=0
    return num
    
# from list of languages, return list of tuples (LANGUAGE, LANGUAGE_NUM_SPEAKERS) for mp3getter 
# ignoring languages less than 50 speakers
# these languages will be used for the experiment
def experiment_languages(languages):
    formatted_languages = []
    for language in languages:
        num = get_num(language)
        if num >= 50:
            formatted_languages.append((language,num))
    return formatted_languages
    
def get_speaker_info(start, stop):

    '''
    Inputs: two integers, corresponding to min and max speaker id number per language
    Outputs: Pandas Dataframe containing speaker filename, birthplace, native_language, age, sex, age_onset of English
    '''

    user_data = []
    for num in range(start,stop):
        info = {'speaker_ID': num, 'filename': 0, 'birthplace':1, 'native_language': 2, 'age':3, 'sex':4, 'age of Eng_onset':5}
        url = "http://accent.gmu.edu/browse_language.php?function=detail&speakerid={}".format(num)
        html = get(url)
        soup = BeautifulSoup(html.content, 'html.parser')
        body = soup.find_all('div', attrs={'class': 'content'})
        try:
            info['filename']=str(body[0].find('h5').text.split()[0])
            bio_bar = soup.find_all('ul', attrs={'class':'bio'})
            info['birthplace'] = str(bio_bar[0].find_all('li')[0].text)[13:-6]
            info['native_language'] = str(bio_bar[0].find_all('li')[1].text.split()[2])
            info['age'] = float(bio_bar[0].find_all('li')[3].text.split()[2].strip(','))
            info['sex'] = str(bio_bar[0].find_all('li')[3].text.split()[3].strip())
            info['age of Eng_onset'] = float(bio_bar[0].find_all('li')[4].text.split()[4].strip())
            user_data.append(info)
        except:
            info['filename'] = ''
            info['birthplace'] = ''
            info['native_language'] = ''
            info['age'] = ''
            info['sex'] = ''
            info['age of Eng_onset'] = ''
            user_data.append(info)
        df = pd.DataFrame(user_data)
        df.to_csv('speaker information.csv')
    return df

# from the accent.gmu website, pass in list of languages to scrape mp3 files and save them to disk
def mp3_getter(lst):
    for i in range(len(lst)):
        for j in range(1,lst[i][1]+1):
            try:
                urllib.request.urlretrieve("http://accent.gmu.edu/soundtracks/{0}{1}.mp3".format(lst[i][0], j), 'C:/git/download/Accented speech recognition/Accent-Classifier/mp3/{0}{1}.mp3'.format(lst[i][0], j))
                print("downloaded")
            except urllib.error.HTTPError:
                pass

# copying mp3 files to wav files 
def mp3_to_wav():
    input_path="C:/git/download/Accented speech recognition/Accent-Classifier/data/mp3"
    output_path="C:/git/download/Accented speech recognition/Accent-Classifier/data/wav"
    
    mp3_list=os.listdir(input_path)

    for i in range(len(mp3_list)):
        wav=AudioSegment.from_mp3("{0}/{1}".format(input_path,mp3_list[i]))
        wav.export("{0}/{1}.wav".format(output_path,mp3_list[i].split(".")[0]),format="wav")
        print("moved")


if __name__ == '__main__':
    lang_list=get_languages()
    #url_list=get_language_urls(lang_list)
    explang_list=experiment_languages(lang_list)
    print(explang_list)
'''
    total_num=0
    for i in range(len(lang_list)):
        speaker_num=get_num(lang_list[i])
        total_num+=speaker_num
        print(speaker_num)
    print("total",total_num)
    get_speaker_info(1,total_num+1)
'''
    #Caution: it takes a long time
    #mp3_getter(formlang_list)
    #mp3_to_wav()
    
    

    
    

