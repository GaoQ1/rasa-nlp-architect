from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
from functools import wraps

import simplejson
import six
from builtins import str
from klein import Klein
from twisted.internet.defer import inlineCallbacks, returnValue
from rasa_nlu_gao import utils, config
from rasa_nlu_gao.utils import json_to_string

from rasa_nlu_api import RasaNluApi
from os import path

logger = logging.getLogger(__name__)

import code

def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse incoming text')

    parser.add_argument('-p', '--port',
                        type=int,
                        default=5000,
                        help='port on which to run server')
    parser.add_argument('-t', '--token',
                        help="auth token. If set, reject requests which don't "
                             "provide this token as a query parameter")
    parser.add_argument('-w', '--write',
                        help='file where logs will be saved')
    parser.add_argument('--cors',
                        nargs="*",
                        help='List of domain patterns from where CORS '
                             '(cross-origin resource sharing) calls are '
                             'allowed. The default value is `[]` which '
                             'forbids all CORS requests.')
    parser.add_argument('-i', '--info',
                        required=True,
                        help="model info path")
    parser.add_argument('-m', '--model',
                        required=True,
                        help="model path")
    
    return parser


def check_cors(f):
    """Wraps a request handler with CORS headers checking."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        origin = request.getHeader('Origin')

        if origin:
            if '*' in self.cors_origins:
                request.setHeader('Access-Control-Allow-Origin', '*')
            elif origin in self.cors_origins:
                request.setHeader('Access-Control-Allow-Origin', origin)
            else:
                request.setResponseCode(403)
                return 'forbidden'

        if request.method.decode('utf-8', 'strict') == 'OPTIONS':
            return ''  # if this is an options call we skip running `f`
        else:
            return f(*args, **kwargs)

    return decorated


def requires_auth(f):
    """Wraps a request handler with token authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        request = args[1]
        if six.PY3:
            token = request.args.get(b'token', [b''])[0].decode("utf8")
        else:
            token = str(request.args.get('token', [''])[0])
        if self.access_token is None or token == self.access_token:
            return f(*args, **kwargs)
        request.setResponseCode(401)
        return 'unauthorized'

    return decorated


def decode_parameters(request):
    """Make sure all the parameters have the same encoding.

    Ensures  py2 / py3 compatibility."""
    return {
        key.decode('utf-8', 'strict'): value[0].decode('utf-8', 'strict')
        for key, value in request.args.items()}


def parameter_or_default(request, name, default=None):
    """Return a parameters value if part of the request, or the default."""

    request_params = decode_parameters(request)
    return request_params.get(name, default)


def dump_to_data_file(data):
    if isinstance(data, six.string_types):
        data_string = data
    else:
        data_string = utils.json_to_string(data)

    return utils.create_temporary_file(data_string, "_training_data")


class RasaNLU(object):
    """Class representing Rasa NLU http server"""

    app = Klein()

    def __init__(self,
                 data_router,
                 loglevel='INFO',
                 logfile=None,
                 token=None,
                 cors_origins=None):

        self._configure_logging(loglevel, logfile)

        self.data_router = data_router
        self.cors_origins = cors_origins if cors_origins else ["*"]
        self.access_token = token


    @staticmethod
    def _load_default_config(path):
        if path:
            return config.load(path).as_dict()
        else:
            return {}

    @staticmethod
    def _configure_logging(loglevel, logfile):
        logging.basicConfig(filename=logfile,
                            level=loglevel)
        logging.captureWarnings(True)

    @app.route("/parse", methods=['GET', 'POST', 'OPTIONS'])
    @requires_auth
    @check_cors
    @inlineCallbacks
    def parse(self, request):
        request.setHeader('Content-Type', 'application/json')
        if request.method.decode('utf-8', 'strict') == 'GET':
            request_params = decode_parameters(request)
        else:
            request_params = simplejson.loads(
                request.content.read().decode('utf-8', 'strict'))

        if 'query' in request_params:
            request_params['q'] = request_params.pop('query')

        if 'q' not in request_params:
            request.setResponseCode(404)
            dumped = json_to_string(
                {"error": "Invalid parse parameter specified"})
            returnValue(dumped)
        else:
            try:
                request.setResponseCode(200)
                response = yield self.data_router.inference(request_params['q'])
                returnValue(json_to_string(response))
            except Exception as e:
                request.setResponseCode(500)
                logger.exception(e)
                returnValue(json_to_string({"error": "{}".format(e)}))


if __name__ == '__main__':
    # Running as standalone python application
    cmdline_args = create_argument_parser().parse_args()

    dir = path.dirname(path.realpath(__file__))
    pretrained_model_info = path.join(dir, cmdline_args.info)
    pretrained_model = path.join(dir, cmdline_args.model)

    rasaApi = RasaNluApi(pretrained_model, pretrained_model_info)
    rasaApi.load_model()

    rasa = RasaNLU(
        rasaApi,
        cmdline_args.write,
        cmdline_args.token,
        cmdline_args.cors,
    )

    logger.info('Started http server on port %s' % cmdline_args.port)
    rasa.app.run('0.0.0.0', cmdline_args.port)
