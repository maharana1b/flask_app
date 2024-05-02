from flask import *

app = Flask(__name__)

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method == "POST":
        user_name = request.form.get('user_name')
        password = request.form.get('pass_word')
        print(user_name,password)
        if user_name == "kore" and password == "mphasis":
            return "Welcome {}".format(user_name)
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html',user_name=user_name)

    

if __name__ =='__main__':
    app.run(debug=True)

