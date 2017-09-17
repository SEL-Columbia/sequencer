import logging
from .NetworkPlan import NetworkPlan
from .Sequencer import Sequencer
 
__version__ = "0.3.2"

logger = logging.getLogger('sequencer')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = \
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# need to set this, otherwise 'root' logger also logs
logging.getLogger('sequencer').propagate = False
