from flask import Flask
from flask import *

app = Flask(__name__)

@app.route('/error')
def error():
    return "<p><strong>Enter correct password</strong></p>"


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/success',methods=["POST"])
def success():
    if request.method == "POST":
        username = request.form.get("user_name")
        password = request.form.get("pass_word")

    if password == 'mphasis':
        resp = make_response(render_template('success.html',username=username))
        resp.set_cookie('username',username)
        return resp
    else:
        return redirect(url_for('error'))




if __name__ =='__main__':
    app.run(debug=True)