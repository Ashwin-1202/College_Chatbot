from flask import Flask, render_template, request, jsonify
from chatbot_core import get_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_input = request.json['message']
        bot_response = get_response(user_input)
        return jsonify({'reply': bot_response})
    except Exception as e:
        return jsonify({'reply': 'Sorry, I encountered an error. Please try again.'})

if __name__ == '__main__':
    print("🎓 College Chatbot Web Server Starting...")
    print("🌐 Open: http://localhost:5000")
    print("⏹️  Press CTRL+C to stop")
    app.run(debug=True, port=5000)