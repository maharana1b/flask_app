from flask import *

app = Flask(__name__)

@app.route('/login', methods=["POST"])
def login():
    user_name = request.form.get('user_name')
    password = request.form.get('pass_word')
    if user_name == "kore" and password == "mphasis":
        return "Welcome {}".format(user_name)
    else:
        return "Invalid credentials. Please try again."

    

if __name__ =='__main__':
    app.run(debug=True)

