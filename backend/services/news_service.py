# Proxy module to keep backward compatibility with tests/imports
import sys
import services.news_service as _orig
# Replace this module object in sys.modules with the original module so imports
# referencing backend.services.news_service and services.news_service point to
# the same module object (monkeypatching will affect both).
sys.modules[__name__] = _orig
