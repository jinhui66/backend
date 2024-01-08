from flask import Flask
from flask import request
from flask import render_template
import torch

app = Flask(__name__)
app.template_folder = 'templates'

@app.route('/',methods=['GET','POST'])
def hello_world():
    return render_template('index.html')

@app.route('/submit',methods=['GET','POST'])
def submit():
    a = request.form.get("a",type=int)
    b = request.form.get("b",type=int)
    # 加载网络
    net = torch.load("net.pth")
    X = torch.tensor([a,b],dtype=torch.float32)
    result = net(X).item()
    # 输出
    return render_template('submit.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/name',methods=['GET','POST'])
# def get_name():
#     if request.method == 'POST':
#         return 'luotuo from POST'
#     else:
#         return 'luotuo from GET'

# @app.route('/fans')
# def get_fans():
#     return '10000'

# @app.route('/userProfile', methods=['GET','POST'])
# def get_profile():
#     if request.method == 'GET':
#         name = request.args.get('name','')
#         print(name)
#         if (name=='luotuo'):        
#             return dict(name='luotuo', fans=10000)
#         else:
#             return dict(name='not luotuo', fans=100)
#     elif request.method == 'POST':
#         print(request.form)
#         print(request.data)
#         print(request.json)
#         name = request.json.get('name')
#         if (name=='luotuo'):        
#             return dict(name='luotuo from POST', fans=10000)
#         else:
#             return dict(name='not luotuo from POST', fans=100)

