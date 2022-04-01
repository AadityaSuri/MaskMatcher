from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_login import UserMixin
import threading
import time
import asyncio
import websockets

#python script to initialize and build the flask app.
#also runs a thread to receive and manage ML predictions

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    # This runs the websockets server that gets the images and the neural network predictions
    @app.before_first_request
    def FINALserver_thread():

        async def hello(websocket):
            name = await websocket.recv()
            print(f"<<< {name}")

            greeting = f"Hello {name}!"

            await websocket.send(greeting)
            print(f">>> {greeting}")

            # This handles getting the images and the prediction from the raspberry pi 
            # and sending whether to lock the door or not back to the pi 
            import globalVars
            while True:
                image = await websocket.recv()
                # print(image)
                print("Got image")

                with open('static/received_file123.png', 'wb') as f:
                    f.write(image)
                    
                prediction = await websocket.recv()
                print(prediction)
                globalVars.prediction = float(prediction)
                print("start thinks that " + prediction)
                
                await websocket.send(str(globalVars.doorLocked))


        async def main():
            async with websockets.serve(hello, "10.93.48.157", 443):
                await asyncio.Future() 


        def FINALserver():
            asyncio.run(main())

        # The websockets server is run as a seperate thread in order to keep it always running
        thread1 = threading.Thread(target=FINALserver)
        thread1.start()


    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True, port=80)
