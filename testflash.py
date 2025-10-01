from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "🎓 College Chatbot is working! If you see this, Flask is running."

if __name__ == '__main__':
    print("Starting Flask server...")
    print("If successful, open: http://localhost:5000")
    app.run(debug=True)