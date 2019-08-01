import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='numberlink-v0',
    entry_point='gym_number_link.envs:NumberLink',
)