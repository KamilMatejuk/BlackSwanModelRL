import uuid6
import connexion
from flask import jsonify
from decouple import Config, RepositoryEnv
from flask_socketio import SocketIO

from model import create_model, save_model, get_best_action, learning_step, get_state
from schemas_request import ActionRequest, LearnRequest


def validate(cls):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                data = cls(**connexion.request.json)
                data.validate()
            except Exception as ex:
                return jsonify({"error": "Invalid request schema", "details": str(ex)}), 401
            return func(data)
        return wrapper
    return decorator


def init():
    try:
        model, optimizer = create_model()
        id = uuid6.uuid7().hex
        save_model(model, optimizer, id)
        return id, 200
    except Exception as ex:
        return jsonify({"error": "Invalid request schema", "details": str(ex)}), 401


@validate(ActionRequest)
def action(data: ActionRequest):
    action = get_best_action(data.id, get_state(data.state))
    return action, 200 # 0 - wait, 1 - switch


@validate(LearnRequest)
def learn(data: LearnRequest):
    loss = learning_step(
        data.id,
        get_state(data.state),
        data.action,
        get_state(data.next_state),
        data.reward)
    return loss, 200


config = Config(RepositoryEnv('.env.local'))
port = config.get('PORT')
app = connexion.FlaskApp(__name__,
        server='tornado',
        specification_dir='',
        options={'swagger_url': '/swagger-ui'})
app.add_api('openapi.yaml')
print(f' * Checkout SwaggerUI http://127.0.0.1:{port}/swagger-ui/')
socketio = SocketIO(app.app)
socketio.run(app.app, port=port)
