import telebot
from telebot import types
import sys
sys.path.append('.')
from gpt.src.generate_from_string import continue_string, get_future_prediction
from deeppavlov import build_model, configs
import numpy as np

print('\n\nLoading model')
BERT = build_model(configs.squad.squad, download=True)

print('\n\n\n\n\n', 30*'---', '\nCreating bot')
token = 'TelegramTokenMANYlettersANDnum34rsXXXXXXX'
bot = telebot.TeleBot(token)


futures = {}


def answer_about_friendship(question):
    FRIEND = """Friendship is a relationship of mutual affection between people.
     Friendship is a stronger form of interpersonal bond than an association. 
     Friendship has been studied in academic fields such as communication, sociology, social psychology, anthropology, and philosophy. 
     various academic theories of friendship have been proposed, including social exchange theory, equity theory, relational dialectics, and attachment styles.
    Although there are many forms of friendship, some of which may vary from place to place, certain characteristics are present in many types of such bonds. 
    Such characteristics include affection; kindness, love, virtue, sympathy, empathy, honesty, altruism, loyalty, 
    generosity, forgiveness, mutual understanding and compassion, enjoyment of each other's company, trust, 
    and the ability to be oneself, express one's feelings to others, and make mistakes without fear of judgment 
    from the friend. Friendship is an essential aspect of relationship building skills.
    """
    return BERT([FRIEND], [question])


def answer_if_confident(question, context, strictness=1.0):
    """
    strictness : bigger the strictness, bigger the threshold
    :return:
    """

    known_ans = "The best cars in the world are Tesla."
    context += known_ans
    known_q = "What are the best cars in the world?"
    absurd = "orange the is go happen furiously?"

    answers, _, (Eq, Ek, Ea) = BERT([context]*3, [question, known_q, absurd])

    threshold = np.sqrt(Ek*Ea)*strictness

    return answers[0] if Eq > threshold else "Neither the tea leaves " \
                                             "nor GPT-2 have found an answer to your question... " \
                                             "I hope you eventually find out what you are searching for."


def extracted_answers(questions, context):
    return {q: answer_if_confident(q, context) for q in questions}


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Hello {}, I was waiting for you. ".format(message.chat.first_name) +
                                      "Be careful with the vase.\n" +
                                      " I am listening to your future. " +
                                      "Wait for a bit and I will come back with something for you.")
    futures[message.chat.id] = get_future_prediction(message.chat.first_name)

    markup = types.ReplyKeyboardMarkup(row_width=1)
    itembtn1 = types.KeyboardButton('Please show me my whole future.')
    markup.add(itembtn1)

    bot.send_message(message.chat.id, 'You can ask me anything now, '
                     'or you can try to dive all by yourself in the messages from your future.',
                     reply_markup=markup)


@bot.message_handler(regexp="Please show me my whole future.")
def show_future(message):
    if message.chat.id in futures:
        bot.send_message(message.chat.id, 'Behold your future: ' + futures[message.chat.id])


@bot.message_handler(regexp="test")
def echo_all(message):
    bot.reply_to(message, 'Are you TESTING ME?')


@bot.message_handler(func=lambda m: True)
def echo_all(message):
    if message.chat.id in futures:
        bot.send_message(message.chat.id, answer_if_confident(message.text, futures[message.chat.id]))


print('Start handling')
bot.polling()