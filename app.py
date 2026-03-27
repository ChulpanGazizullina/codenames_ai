from flask import Flask, jsonify, session, render_template, request
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import os
import random
from itertools import combinations
import zipfile
import urllib.request

app = Flask(__name__)
app.secret_key = os.urandom(24)

URL = "https://vectors.nlpl.eu/repository/20/180.zip"
ZIP_PATH = "model.zip"
MODEL_PATH = "model.bin"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

v2v_model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)


df_nouns = pd.read_csv("data/nouns.csv")



def sorting_out_words(groupsize: int) -> tuple[str, float, list[str]]:
    """функция перебора наборов слов из списка green_ai (размер набора -- параметр groupsize)
    и нахождения оптимальной подсказки, близкой к обоим словам и далекой от чёрных и оранжевых слов.
    Для каждого набора размером groupsize:
    1. векторизует каждое слово набора
    2. считает средний вектор этих слов
    3. берет 20 ближайших кандидатов на подсказку из модели к полученноему серднему вектору
    4. проверяет, что кандидат не является словом из поля, а также подсказкой, которая была дана ранее
    5. считает score кандидата-подсказки по формуле: benefit - 2.5 * fall_black - fall_orange, где
    benefit -- средняя косинусная близость кандидата-подсказки ко всем словам набора
    fall_black -- максимальная косинусная близость к черному слову
    fall_orange -- максимальная косинусная близость к оранжевому слову
    проверяя каждый набор размером groupsize, функция находит лучшую подсказку с максимальным score
    возвращает best_score лучший (наибольший) score
    best_cand лучшую подсказку
    hidden_words загаданные слова"""
    best_score = (
        -4
    )  # я решила, что это минимальный возможный скор на моей функции, проверить математику
    best_cand = None
    hidden_words = []
    green_ai = session.get("green_ai")
    black = session.get("black")  # тут всего 2 слова
    all_hints_backend = session.get('all_hints_backend')
    for sample in combinations(green_ai, groupsize):
        vectors = np.array([v2v_model[f"{w}_NOUN"] for w in sample])
        # моя модель училась на НКРЯ, поэтому я уверена, что частотные существительные в этой модели есть.
        # Не считаю нужным проверять, есть ли слово в модели
        mean_vector = np.mean(vectors, axis=0)
        candidates = v2v_model.similar_by_vector(mean_vector, topn=20)
        for cand, _ in candidates:  # cand у меня вида "слово_NOUN"
            if "_NOUN" not in cand:  # хочу подсказки only существительные
                continue
            if cand.split("_")[0] in session.get(
                "words"
            ):  # проверка, что подсказка не входит в слово на поле
                continue
            if (
                cand.split("_")[0] in all_hints_backend
            ):  # проверка, что слово не было среди подсказок раньше
                continue
            benefit = np.mean([v2v_model.similarity(f"{w}_NOUN", cand) for w in sample])
            fall_black = max(
                v2v_model.similarity(f"{black[0]}_NOUN", cand),
                v2v_model.similarity(f"{black[1]}_NOUN", cand),
            )
            # потому что черных всего два слова по условию
            fall_orange = (
                -1
            )  # ищем самое близкое к кандидату оранжевое слово и штрафуем за него
            for orange in session.get("left"):
                current_orange = v2v_model.similarity(f"{orange}_NOUN", cand)
                if current_orange > fall_orange:
                    fall_orange = current_orange

            score = (
                benefit - 2.5 * fall_black - fall_orange
            )  # функцию придумала сама по вайбику, суть в том, что за близость к черному слову штрафую сильно,
            # за близость к оранжевому штрафую меньше

            if score > best_score:
                best_score = score  # наибольший полученнный score
                best_cand = cand.split("_")[0]  # подсказка, которая имеет best_score
                hidden_words = [
                    w for w in sample
                ]  # слова, которые загаданы подсказкой best_cand
    return best_cand, best_score, hidden_words


@app.route("/")
def start():
    return render_template("index.html")


@app.route("/play")
def play():
    """функция генерирует поле из слов 5х5
    выбирает рандомно 15 слов, которые будут загаданы
    2 черных слова, которые выбирать нельзя
    оставшиеся 8 слов -- нейтральные"""
    wordlist = df_nouns["word_0"].sample(n=25).tolist()
    session["words"] = wordlist  # все слова поля
    session["green_ai"] = random.sample(
        wordlist, 15
    )  # зеленые слова, которые нужно угадать
    session["black"] = random.sample(
        list(set(wordlist) - set(session.get("green_ai"))), 2
    )  # чёрные слова, которые угадывать нельзя
    session["left"] = list(
        set(wordlist) - set(session.get("green_ai")) - set(session.get("black"))
    )  # оранжевые слова, нейтральные
    words_left = len(session.get("green_ai"))
    all_hints_frontend = []
    all_hints_backend = []
    session['all_hints_frontend'] = all_hints_frontend
    session['all_hints_backend'] = all_hints_backend
    return render_template("play.html", words=wordlist, words_left=words_left)


@app.route("/generate_hint")
def generate():
    """параметр groupsize работает так, что:
    1. запрещает, чтобы на поле осталось одно слово (=если осталось 3, то возьмет максимум)
    2. равен 2 или 3, если п.1 выполняется (выбор между 2 и 3 происходит в функции sorting_out_words)
    """
    green_ai = session.get("green_ai")
    all_hints_frontend = session.get('all_hints_frontend')
    all_hints_backend = session.get('all_hints_backend')
    result = []
    if len(green_ai) == 2 or len(green_ai) == 4:
        result.append(sorting_out_words(2))

    if len(green_ai) == 3:
        result.append(sorting_out_words(3))

    else:
        result = []
        for groupsize in [2, 3]:
            result.append(sorting_out_words(groupsize))

    best_cand, best_score, hidden_words = max(result, key=lambda x: x[1])
    print("best_cand", best_cand)
    print("best_score", best_score)
    print("hidden_words", hidden_words)
    all_hints_frontend.append(f" {best_cand} ({len(hidden_words)} слова)")
    all_hints_backend.append(best_cand)
    session['all_hints_frontend'] = all_hints_frontend
    session['all_hints_backend'] = all_hints_backend
    print(session.get('all_hints_backend'))
    print(session.get('all_hints_frontend'))   
    # это тот список всех подсказок, который будет выводиться на экран
    # этот список будет использоваться для проверки слова: выдавалось ли оно уже в качестве подсказки
    return jsonify(
        {
            "hint": best_cand,
            "n_words": len(hidden_words),
            "all_hints": session.get('all_hints_frontend'),
        }
    )


@app.route("/checkword", methods=["POST"])
def checkword():
    """получает на вход слово, которое пользователь нажал
    проверяет, является ли это слово
    1. зеленым (игра продолжается)
    2. оранжевым (игра продолжается)
    3. черным (игра закончена, пользователь проиграл)
    4. последним зеленым словом (больше зеленых слов нет) (пользователь выиграл)"""
    game_over = False
    message = ""
    data = request.json
    word = data.get("word")
    black = session.get("black", [])
    green_ai = session.get("green_ai", [])
    words_left = len(session.get("green_ai"))
    if word in green_ai:
        result = "green"
        green_ai.remove(
            word
        )  # чтобы при генерировании следущей подсказки угаданное слово не учитывалось
        session["green_ai"] = green_ai
        words_left = len(green_ai)
        if len(green_ai) == 0:
            game_over = True
            message = "Вы умеете думать про ассоциации так же, как это делает нейронка! Поздравляем, вы выиграли!"

    elif word in black:
        result = "black"
        game_over = True
        message = "Вы выбрали черное слово, увы! Игра окончена"
    else:
        result = "orange"

    print(
        session["green_ai"]
    )  # вывести в терминал, проверить, удалилось ли выбранное зеленое слово из всех,
    # для которых будет даваться подсказка

    return jsonify(
        {
            "word": word,
            "result": result,
            "game_over": game_over,
            "message": message,
            "words_left": words_left,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
