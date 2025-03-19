import logging

# Silence matplotlib debug logs

logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
