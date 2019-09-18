from flask import Flask
app = Flask(__name__)

@app.route('/') # Flask类的route()函数是一个装饰器，它告诉应用程序哪个URL应该调用相关的函数。
def hello_world():
   return 'Hello World11122'

if __name__ == '__main__':
   app.run(host='127.0.0.1', port=8080, debug=True)