# Proxy module to keep backward compatibility with tests/imports
import sys
import services.market_data as _orig
# Replace this module object in sys.modules with the original module so imports
# referencing backend.services.market_data and services.market_data point to
# the same module object.
sys.modules[__name__] = _orig
