import logging
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import io

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, CallbackQueryHandler, filters
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.io as pio

nltk.download('stopwords')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TOKEN = '7169787095:AAHDGZWRncRFV7-ltAC-x0_qtTDEawNr2-c'

# --- Предобработка и обучение модели ---


def preprocess_text(text):
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [stemmer.stem(
        word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Демо данные
texts = [
    "Artificial intelligence and machine learning are transforming the world.",
    "The economy is showing signs of recovery after the pandemic.",
    "The new movie was a blockbuster hit and received positive reviews.",
    "Scientists have discovered a new species in the Amazon rainforest.",
    "Climate change is one of the biggest challenges facing humanity.",
    "Input the neural network to analyze the data system algorithm.",
    "Generate output layers with convolutional neural networks architecture.",
    "The transformer model utilizes attention mechanisms for processing.",
    "Compute the gradients and update the weights in the backpropagation.",
    "The AI is intelligent and can mimic human-like text generation."
]

labels = ['human', 'human', 'human', 'human', 'human',
          'ai', 'ai', 'ai', 'ai', 'ai']

data = pd.DataFrame({'text': texts, 'label': labels})
data['processed_text'] = data['text'].apply(preprocess_text)

X = data['processed_text']
y = data['label']

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Точность модели: {accuracy * 100:.2f}%")

# --- Бот ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton('Помощь', callback_data='help'),
            InlineKeyboardButton('О боте', callback_data='about')
        ],
        [
            InlineKeyboardButton('Показать 3D график', callback_data='show_3d_graph')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "👋 Привет! Я бот, который определяет, кем написан текст: человеком или искусственным интеллектом.\n\n"
        "Отправь мне любой текст на английском языке, и я проанализирую его.",
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(
            "📝 *Инструкция по использованию бота:*\n\n"
            "1. Отправьте мне любой текст на английском языке.\n"
            "2. Я проанализирую его и сообщу, кем он вероятнее всего написан: человеком или искусственным интеллектом.\n"
            "3. Вы можете просмотреть 3D график, связанный с анализом данных.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            "📝 *Инструкция по использованию бота:*\n\n"
            "1. Отправьте мне любой текст на английском языке.\n"
            "2. Я проанализирую его и сообщу, кем он вероятнее всего написан: человеком или искусственным интеллектом.\n"
            "3. Вы можете просмотреть 3D график, связанный с анализом данных.",
            parse_mode='Markdown'
        )


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(
            "🤖 *О боте:*\n\n"
            "Я использую алгоритмы машинного обучения для анализа текстов и определения их происхождения.\n"
            "Создан для демонстрации возможностей NLP и машинного обучения в Telegram Bot API.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            "🤖 *О боте:*\n\n"
            "Я использую алгоритмы машинного обучения для анализа текстов и определения их происхождения.\n"
            "Создан для демонстрации возможностей NLP и машинного обучения в Telegram Bot API.",
            parse_mode='Markdown'
        )


async def send_3d_graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width',
                        z='petal_length', color='species')
    fig.update_layout(title='3D Scatter Plot')

    graph_filename = '3d_graph.html'
    pio.write_html(fig, file=graph_filename, auto_open=False)

    with open(graph_filename, 'rb') as f:
        await update.callback_query.message.reply_document(document=f, filename=graph_filename)


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data == 'help':
        await help_command(update, context)
    elif data == 'about':
        await about_command(update, context)
    elif data == 'show_3d_graph':
        await send_3d_graph(update, context)
    else:
        await query.answer()


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    processed_text = preprocess_text(user_text)
    text_vector = vectorizer.transform([processed_text])

    prediction = model.predict(text_vector)[0]

    if prediction == 'human':
        response = "✍️ Этот текст, вероятно, написан *человеком*."
    else:
        response = "🤖 Этот текст, вероятно, сгенерирован *искусственным интеллектом*."

    await update.message.reply_text(response, parse_mode='Markdown')

    user_data = context.user_data
    user_data['analysis'] = user_data.get('analysis', [])
    user_data['analysis'].append(prediction)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data
    analysis = user_data.get('analysis', [])

    if not analysis:
        await update.message.reply_text("Вы ещё не анализировали тексты.")
        return

    human_count = analysis.count('human')
    ai_count = analysis.count('ai')

    labels = ['Человек', 'ИИ']
    counts = [human_count, ai_count]

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['skyblue', 'salmon'])
    ax.set_title('Статистика анализов')
    ax.set_ylabel('Количество')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    await update.message.reply_photo(photo=buffer)

def main():
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('about', about_command))
    application.add_handler(CommandHandler('stats', stats_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, analyze_text))

    application.run_polling()


if __name__ == '__main__':
    main()
