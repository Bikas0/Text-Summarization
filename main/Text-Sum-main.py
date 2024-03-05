# python == 3.11.7
import re
import nltk
import torch
from pathlib import Path
# Define the device if using GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import pipeline
# from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

tokenizer = AutoTokenizer.from_pretrained(Path("/home/bikas/Text-Summarization/main/summary/tokenizer"))
model = "/home/bikas/Text-Summarization/main/summary/pegasus-samsum-model"
def remove_spaces_before_punctuation(text):
    pattern = re.compile(r'(\s+)([.,;!?])')
    result = pattern.sub(r'\2', text)
    result = re.sub(r'\[|\]', '', result)
    return result


def replace_pronouns(text):
    # Replace "they" with "he" or "she" based on context
    text = re.sub(r'\bthey\b', 'He/She', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(are|have|were)\b', lambda x: {'are': 'is', 'have': 'has', 'were': 'was'}[x.group()], text)
    return text


def clean_and_lemmatize(text):
    # Remove digits, symbols, punctuation marks, and newline characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s,-]', '', text.replace('\n', ''))
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize each token and join back into a sentence
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_text


# input_text = input("\nGive me Your Full Month Log hour Task Description within 5500 Characters: \n")
input_text = """
Video
Gallery
ePaper
Archive
Bangla
 
Tuesday, 5 March, 2024, 11:31 AM
logo
Advance Search
 
2023
 
home
PRINT EDITION
Front Page
Back Page
City News
Editorial
Op-Ed
Foreign News
Business
Sports
Countryside
FEATURE
Women's Own
Book Review
Literature
Life & Style
Observer TeCH
Law & Justice
Health & Nutrition
Young Observer
ELECTION 2024
BUSINESS
NATIONAL
Politics
Crime & Court
INTERNATIONAL
EDUCATION
DON'T MISS
SPORTS
COUNTRYSIDE
HEALTH
SPECIAL ISSUE
World Cup 2022
Magazine 2019
Magazine 2017
Magazine 2016
Magazine 2015
VISUAL
Home 
UK, EU eye customs deal on Channel migrants crossings
Published : Tuesday, 5 March, 2024 at 9:46 AM  Count : 136
Observer Online Desk
Submit   
facebook sharing buttontwitter sharing buttonlinkedin sharing buttonblogger sharing buttonpinterest sharing buttonemail sharing buttonsharethis sharing button

UK, EU eye customs deal on Channel migrants crossings
UK, EU eye customs deal on Channel migrants crossings


A grouping of northern European countries on Monday agreed to work on a new "customs partnership" to disrupt the supply of small boats used to carry migrants cross the Channel, Britain's interior ministry said.

The UK and France will lead on setting up the new initiative, which will see countries' customs agencies share information on the shipping of small boat materials more effectively, it added.

It comes as Britain, no longer an EU member since 2020, tries to stem the flow of tens of thousands of migrants arriving each year on its southeastern shores on small boats from mainland Europe, AFP reports.

The journeys have repeatedly proved deadly, with the latest victim a seven-year-old girl who drowned on Sunday when a small boat carrying 16 migrants heading from northern France to Britain capsized.

UK Prime Minister Rishi Sunak vowed at the start of last year to "stop the boats" but nearly 30,000 still made the crossing in 2023 despite stepped up efforts to thwart them.

The issue -- a politically potent one given the UK government's promise to "take back control" of the country's borders after Brexit -- is set to feature prominently in a general election later this year.

The plan for better customs coordination was discussed at a ministerial meeting Monday of the so-called Calais Group in Brussels.

It comprises the UK, France, Belgium, Germany and the Netherlands, as well as the European Commission and its agencies, and works to promote cooperation on tackling irregular migration.

"This is an initiative to work with countries throughout the supply chain of small boat materials, and will build on the effective work already being done to prevent small boat launches from northern France," the UK interior ministry said in a statement.

"Partnership countries and their customs agencies will... be able to share information more effectively to disrupt shipments of small boat materials, preventing them from making it to the English Channel."

The grouping is set to discuss the plan again at its next meeting in April.

Monday's gathering also explored working with social media companies to tackle online activity by people-smuggling networks, the UK ministry said.

In addition, participants discussed a recent UK deal with Frontex, the European border and coastguard agency, to exchange information and intelligence and take on the gangs together, it added.

SR
Related Topics
UK   EU   France   Belgium   Germany   Netherlands  


facebook sharing buttontwitter sharing buttonlinkedin sharing buttonblogger sharing buttonpinterest sharing buttonemail sharing buttonsharethis sharing button



« Previous		Next »

You Might Also Like


US slaps fresh sanctions on Zimbabwe President, other leaders
UK, EU eye customs deal on Channel migrants crossings
UN team says rape, gang rape likely occurred during Hamas attack on Israel
Children starving to death in Gaza hospitals: WHO
Four arrested over raping Spanish blogger in India
Shehbaz Sharif sworn in as Pakistan’s new prime minister
Israel demolishes Palestinian's home in West Bank
Nikki Haley beats Trump in Washington DC for 1st primary victory


Latest News
Uncle-nephew killed in Faridpur road mishap
US slaps fresh sanctions on Zimbabwe President, other leaders
MoU signed between UniSZA Malaysia, IIUC
RMG workers' protest block Dhaka-Mymensingh highway
UK, EU eye customs deal on Channel migrants crossings
UN team says rape, gang rape likely occurred during Hamas attack on Israel
Children starving to death in Gaza hospitals: WHO
Britain hoping to host 2029 athletics world championship
Two multi-storied buildings in Dhanmandi declared 'Risky'
Arafat for ensuring transparency in disbursing grants for films
Most Read News
34 held in crackdown at restaurants in Dhaka
Polish national found dead in Ctg hotel
Warrant against Evaly's Rassel, wife issued in 3 cases
High-level probe body forms over Bailey Road fire
12 restaurants at Dhanmondi's Gawsia Twin Peak sealed off
HC rules for compensating Bailey Road fire victim families
Financial Literacy Day focuses on customers’ awareness
Bangladesh can take advantage of EU’s shortage of IT specialists
Medical student shot by teacher in Sirajganj
Bangladesh bears the brunt of Rohingya crisis

Editor : Iqbal Sobhan Chowdhury
Published by the Editor on behalf of the Observer Ltd. from Globe Printers, 24/A, New Eskaton Road, Ramna, Dhaka.
Editorial, News and Commercial Offices : Aziz Bhaban (2nd floor), 93, Motijheel C/A, Dhaka-1000.
Phone: PABX- 41053001-06; Online: 41053014; Advertisement: 41053012.
E-mail: info©dailyobserverbd.com, news©dailyobserverbd.com, advertisement©dailyobserverbd.com, For Online Edition: mailobserverbd©gmail.com
  [ABOUT US]     [CONTACT US]   [AD RATE]   Developed & Maintenance by i2soft
"""

# Tokenize the input text
tokens_org_text = tokenizer.tokenize(input_text)
# Get the length of the tokenized sequence
sequence_length_org_text = len(tokens_org_text)

print(sequence_length_org_text)

input_text = clean_and_lemmatize(input_text)
# Tokenize the input text
tokens = tokenizer.tokenize(input_text)

# Get the length of the tokenized sequence
sequence_length = len(tokens)
print(sequence_length)


if sequence_length >= 1024:
    print(f"Your Text token length is {sequence_length_org_text} Which exceeds 1023 tokens")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
   
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    # Example usage
    text = pipe(input_text, **gen_kwargs)[0]["summary_text"]
    output_text = replace_pronouns(remove_spaces_before_punctuation(text))
    print("Summary: ", output_text)
    # Clear the GPU cache
    torch.cuda.empty_cache()
