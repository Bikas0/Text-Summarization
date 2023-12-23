# python == 3.11.7
import re
import nltk
from transformers import pipeline
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


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
input_text = """['investigate feedback for eb app add task, meeting, and call log in lead show task, meeting, and call log from lead show lead and call log from account detail add lead, call log, and meeting from account detail crt screen fix bug in adding lead from account test and fix attendance in eb app rd business card scanner in eb app fix email issue in call detail add account in dm and lead list rd on business card scanner meeting on eb app feedback fix bug in lead add add image cropper in card scanner change attendance ui fix attendance design and fix activity log add selected item bug add filter in colleague screen and fix activity add task offspring add vendor address in colleague list investigate all colleague list not found issue add vendor address in colleague list jam all colleague not issue add month stepper attendance add month picker in attendance fix loading problem in monthly attendance on phone number format rd daily activity setup laptop for flutter development change phone number format algorithm fix loading issue in daily attendance screen add daily activity summary design profile screen change design in detail screen release build aab reload contact colleague and account using refresh indicator add doughnut for attendance percent meeting on eb app not found issue in play store add address for partner in colleague list and change dashboard order designation in colleague and contact call detail move favorite button position and phone state stream bug eb app feedback collection and analysis change search algorithm add contact share feature build aab for play store meeting with md sir on new requirement check in for eb app papers requirement for check in feature share contact using or vcard build aab for play store save contact analyze check in feature in eb sort client by name or time sort account by name and time add notice, my team, and favorite in home screen sort colleague by name or update time fix search issue in colleague disable icon if phone or email not found in contact detail page share and save colleague phone and email add time and in meeting detail fix app close issue when permission is denied fix favorite height issue fix design issue fix design in attendance screen meeting with md sir on check - in feature show device name in contact modify filter in colleague design fix add monthly lumber hour pie chart rd on theme in flutter to change default color of button fix device call log issue design and bug fixing fix issue in disabling email icon and fix issue in device contact name show integrate favorite add api sync favorite with server design and bug fixing change monthly activity graph fix bug in opt timer show colleague image from eb profile pic bug fixing in lead add show people image in contact list and call detail compute attendance work hour design fix document api for business card upload study wetucom rd']
"""
Org_len = len(input_text)
input_text = clean_and_lemmatize(input_text)

if len(input_text) > 5500:
    print(f"Your Text characters number is {Org_len} Which exceeds 5500 Characters")
else:
    tokenizer = AutoTokenizer.from_pretrained("summary/tokenizer")
    # model = "pegasus-samsum-model"
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

    pipe = pipeline("summarization", model="summary/pegasus-samsum-model", tokenizer=tokenizer)
    # Example usage
    text = pipe(input_text, **gen_kwargs)[0]["summary_text"]
    output_text = replace_pronouns(remove_spaces_before_punctuation(text))
    print("Summary: ", output_text)