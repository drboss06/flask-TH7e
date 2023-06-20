import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Определение архитектуры нейронной сети
def addition_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_shape=(2,), activation='relu'),  # Полносвязный слой с 64 нейронами
        tf.keras.layers.Dense(64, activation='relu'),  # Дополнительный полносвязный слой с 64 нейронами
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Выходной слой с одним нейроном
    ])
    return model

def make_model():
# Создание модели
    model = addition_model()

    # Компиляция модели с функцией потерь и оптимизатором
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Подготовка обучающих данных
    train_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    train_Y = [0, 1, 1, 2]

    # Обучение модели
    model.fit(train_X, train_Y, epochs=100, verbose=1)

    # Проверка работы модели
    test_X = [[0, 2], [1, 3], [2, 2]]
    predictions = model.predict(test_X)
    print(predictions)

    test_result = model.predict([[0, 2], [1, 3], [2, 2]])

    print(test_result)

    model.save('addition_model.h5')

model = tf.keras.models.load_model('addition_model.h5')

app = Flask(__name__)
CORS(app, resources={r"/add": {"origins": "*"}})

@app.route('/main')
def index():
    return render_template('Bootindex.html')

@app.route('/main/old')
def old_index():
    return render_template('index.html')


@app.route('/add', methods=['POST'])
def add_numbers():
    if request.method == 'OPTIONS':
        # Обработка предварительного запроса (preflight request) для CORS
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        numbers = data['numbers']

        # Преобразование чисел в numpy массив
        inputs = tf.convert_to_tensor([numbers], dtype=tf.float32)

        # Вычисление предсказания с использованием модели
        result = model.predict(inputs)

        # Преобразование значения в тип float и возвращение в JSON-ответе
        result_float = float(result[0][0])

        return jsonify({'result': round(result_float, 0), 'result_float': result_float}), 200

    except Exception as e:
        #traceback.print_exc()  # Вывод трассировки стека исключения
        return jsonify({'error': str(e)}), 500

def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST')
    return response

if __name__ == '__main__':
    app.run()
