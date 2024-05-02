from flask import Flask
from flask import *



app = Flask(__name__)

# @app.route('/')
# def home():
#     return "First flask app"


# @app.route('/home/<name>')
# def name(name):
#     return "hello: " + name

# @app.route('/age/<int:age>')
# def age(age):
#     return "Age = {}".format(age)

# def about():
#     return "This is about page"

# app.add_url_rule("/about","about",about)


@app.route('/admin')
def admin():
    return "This is admin page"

@app.route('/library')
def library():
    return "This is libray page"

@app.route("/student")
def student():
    return "This is Student Page"


@app.route("/user/<name>")
def user(name):
    if name == "admin":
        return redirect(url_for('admin'))
    if name == "library":
        return redirect(url_for('library'))
    if name == "student":
        return redirect(url_for('student'))
    

if __name__ == '__main__':
    app.run(debug=True)

