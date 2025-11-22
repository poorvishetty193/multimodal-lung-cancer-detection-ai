from loguru import logger
logger.add("/app/logs/api.log", rotation="10 MB", retention="10 days", level="INFO")
